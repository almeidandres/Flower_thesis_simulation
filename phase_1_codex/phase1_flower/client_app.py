"""ClientApp for Phase 1 federated workloads."""

from __future__ import annotations

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .data import load_partition_data
from .models import build_model
from .task import evaluate_model, get_device, train_model

app = ClientApp()


def _resolve_client_partition(context: Context) -> tuple[int, int]:
    cid = int(context.node_config.get("partition-id", context.node_id % 10))
    num_clients = int(context.node_config.get("num-partitions", context.run_config["num-clients"]))
    return cid, num_clients


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train selected client and return updated weights + metrics."""
    cfg = msg.content["config"]
    task_name = str(cfg["task"])

    model = build_model(task_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    cid, num_clients = _resolve_client_partition(context)
    batch_size = int(context.run_config["batch-size"])
    _ufd = context.run_config["use-fake-data"]
    use_fake_data = _ufd if isinstance(_ufd, bool) else str(_ufd).lower() == "true"
    train_samples_per_client = int(context.run_config["train-samples-per-client"])
    eval_samples_per_client = int(context.run_config["eval-samples-per-client"])

    train_loader, _ = load_partition_data(
        task_name=task_name,
        cid=cid,
        num_clients=num_clients,
        batch_size=batch_size,
        use_fake_data=use_fake_data,
        train_samples_per_client=train_samples_per_client,
        eval_samples_per_client=eval_samples_per_client,
    )

    train_loss, train_acc = train_model(
        model=model,
        train_loader=train_loader,
        local_epochs=int(cfg["local-epochs"]),
        lr=float(cfg["lr"]),
        device=device,
    )

    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(
                {
                    "num-examples": len(train_loader.dataset),
                    "train-loss": train_loss,
                    "train-accuracy": train_acc,
                }
            ),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate client model and return metrics only."""
    cfg = msg.content["config"]
    task_name = str(cfg["task"])

    model = build_model(task_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    cid, num_clients = _resolve_client_partition(context)
    batch_size = int(context.run_config["batch-size"])
    _ufd = context.run_config["use-fake-data"]
    use_fake_data = _ufd if isinstance(_ufd, bool) else str(_ufd).lower() == "true"
    train_samples_per_client = int(context.run_config["train-samples-per-client"])
    eval_samples_per_client = int(context.run_config["eval-samples-per-client"])

    _, eval_loader = load_partition_data(
        task_name=task_name,
        cid=cid,
        num_clients=num_clients,
        batch_size=batch_size,
        use_fake_data=use_fake_data,
        train_samples_per_client=train_samples_per_client,
        eval_samples_per_client=eval_samples_per_client,
    )

    eval_loss, eval_acc = evaluate_model(model, eval_loader, device)
    content = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "num-examples": len(eval_loader.dataset),
                    "eval-loss": eval_loss,
                    "eval-accuracy": eval_acc,
                }
            )
        }
    )
    return Message(content=content, reply_to=msg)
