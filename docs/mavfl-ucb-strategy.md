# UCB-Based Vehicle Selection Strategy — MAVFL Paper Reference

Source: *Mobility-Aware Federated Learning: Multi-Armed Bandit Based Selection in Vehicular Network*
DOI: https://doi.org/10.48550/arXiv.2410.10451

---

## 1. Motivation & Core Insight

The selection strategy exists because future vehicle locations are unknown at selection time, making direct optimization of the utility function (problem P1) intractable. The MAB framework is used as the online solution: vehicles are treated as "arms", and the UCB policy balances exploiting well-performing vehicles with exploring under-tried ones.

The key insight from convergence analysis (Theorem 1, Remark 1): **the real-time successful training ratio `p^r` is the dominant factor in convergence speed**. Smart vehicle selection raises `p^r` by picking vehicles less likely to drive out of coverage, which directly reduces training loss.

---

## 2. FL System Model (Selection Context)

There are `K_0` vehicles total. Each round `r ∈ {1, …, R}` proceeds in four phases:

1. **Selection & distribution**: Server picks set `S^r` of vehicles and pushes global model `w^r`.
   ```
   w^{r,0}_k ← w^r,   ∀k: x^{r,0}_k ∈ X_s,  k ∈ S^r
   ```
2. **Local SGD**: Each selected vehicle runs `E` gradient descent epochs:
   ```
   g^{r,e}_k  ←  (1/|D_k|) Σ_{i∈D_k} ∇ℓ(w^{r,e}_k; z^i_k)
   w^{r,e+1}_k ← w^{r,e}_k - η * g^{r,e}_k,   e ∈ [0, E-1]
   ```
3. **Upload attempt**: Vehicle `k` tries to upload accumulated gradient `g^r_k = Σ_{e=0}^{E-1} g^{r,e}_k`.
   If vehicle drove outside coverage segment during training:
   ```
   1^r_k = 1   if  x^{r,E}_k ∈ X_s   (still in segment → upload succeeds)
   1^r_k = 0   if  x^{r,E}_k ∉ X_s   (drove out → upload fails / dropout)
   ```
4. **Aggregation**: Server aggregates all received updates:
   ```
   w^{r+1} = w^r - η * Σ_k 1^r_k * (g^r_k / Σ_k 1^r_k),   if Σ_k 1^r_k ≠ 0
   w^{r+1} = w^r                                              if Σ_k 1^r_k = 0
   ```

---

## 3. Successful Training Ratio

After each round, the server observes:

```
N^r = ∪_{k: 1^r_k = 1} {k}          (set of vehicles whose updates were received)

p^r = |N^r| / |S^r|                  (successful training ratio, Eq. 5)
```

- `|N^r|` — count of received model updates
- `|S^r|` — count of vehicles that downloaded the global model
- `p^r ∈ [0, 1]`; equals 1 when all vehicles are stationary; equals 0 when all selected vehicles drive out

This ratio is observable in real-time — the server counts successful uploads without needing to know vehicle positions in advance.

---

## 4. Convergence Bound and Why p^r Matters

### Assumptions used in proof

- **A1 (L-smooth)**: `||∇F(w1) - ∇F(w2)|| ≤ L||w1 - w2||`
- **A2 (bounded gradient norm)**: `E[||∇f(w^{r,e}_k)||²] ≤ G²`
- **A3 (bounded mini-batch variance)**: `||g(w) - ∇f(w)||² ≤ σ²`
- **A4 (bounded gradient divergence)**: `(1/K) Σ_k ||∇f(w) - ∇F(w)||² ≤ ε²_g`
- **Lemma 1 (local drift)**: `E[Σ_k ||w̄^t - w^t_k||²] ≤ 4Kη²_t(E-1)²G²`

### Theorem 1 (Convergence Rate)

Let `η_{t0} ≤ 2η_t` for all `t - t0 ≤ E - 1`. Then:

```
(1/T) Σ_{t=1}^{T} E[||∇F(w̄^t)||²]

  ≤ (1/T) Σ_{t=1}^{T} (2 / (η * p^t)) * (E[F(w̄^0)] - F_inf)
    + 2ε²_g
    + η(δ² + G²)L
    + 4η²(E-1)²G²L²
```

Where `w̄^t` is the virtual global model after round `t` and `F_inf` is the infimum of `F`.

### Key takeaway

The term `2 / (η * p^t)` is the only round-varying factor. When `p^t` is low (many vehicles dropped out), the convergence bound is loose (slow convergence). When `p^t ≈ 1` (all selected vehicles upload successfully), the bound tightens. **Maximizing `p^t` across rounds is equivalent to minimizing the convergence bound.**

### Proof sketch (Appendix A)

Starting from the virtual global model update:
```
w̄^{t+1} = w̄^t - η * Σ_k 1^t_k * (g^t_k / Σ_k 1^t_k)
```

Using L-smoothness (A1):
```
E[F(w̄^{t+1})] ≤ E[F(w̄^t)]
    + (η²L/2) * E[||Σ_k (g^t_k 1^t_k / Σ_k 1^t_k)||²]       ... (11, term 2)
    - η * E[⟨∇F(w̄^t), Σ_k (g^t_k 1^t_k / Σ_k 1^t_k)⟩]      ... (11, term 3)
```

**Term 2** bounds the squared norm of the weighted average gradient. Using independence of vehicle mobility and local computing:
```
E[Σ_k (g^t_k 1^t_k / Σ_k 1^t_k)] = p^t * Σ_k ∇f(w^t_k)
```
After expansion and applying A2, A3, A4:
```
E[||Σ_k (g^t_k 1^t_k / Σ_k 1^t_k)||²]
  = E[||Σ_k (g^t_k 1^t_k / Σ_k 1^t_k) - p^t Σ_k ∇f(w^t_k)||²]
    - (p^t)² * Σ_k E[||∇f(w^t_k)||²]
  ≤ p^t * K * (δ² + G²)       ... (12)
```

**Term 3** (inner product with ∇F) expands via polarization identity
`⟨a,b⟩ = (1/2)(||a||² + ||b||² - ||a-b||²)`:
```
-η * E[⟨∇F(w̄^t), Σ_k (g^t_k 1^t_k / Σ_k 1^t_k)⟩]
  = -η * p^t * Σ_k E[⟨∇F(w̄^t), ∇f(w^t_k)⟩]
  = -(ηp^t/2) * Σ_k (E[||∇F(w̄^t)||²] + E[||∇f(w^t_k)||²] - E[||∇F(w̄^t) - ∇f(w^t_k)||²])
```
Applying triangle inequality, L-smoothness, Lemma 1 (local drift bound), and A4:
```
≤ -(ηp^t/2) * Σ_k (E[||∇F(w̄^t)||²] - 4η²(E-1)²G²L² - 2ε²_g)    ... (13)
```

Combining (12) and (13) into (11) and averaging over `T` rounds yields Theorem 1.

---

## 5. Utility Function

To simultaneously minimize training loss and delay, the paper defines a per-round utility:

```
Φ^r(a^r_k) = α * p^r(a^r_k)  -  (1-α) * (T(a^r_k) - T_min) / (T_max - T_min)
```

- `a^r = [a^r_1, …, a^r_K]` — binary vehicle selection vector (`a^r_k = 1` if selected)
- `α ∈ [0, 1]` — balances training quality vs. speed (paper: α = 0.6)
- First term: fraction of selected vehicles that successfully upload (higher = better model quality)
- Second term: normalized round duration (higher T = worse, so subtracted)
- `T_min`, `T_max`: minimum and maximum possible round durations (used for normalization)

### Round Duration Model

Total round duration is the maximum over all selected vehicles of (comm time + compute time):
```
T = T_c + T_p = Σ_{r∈R} max_{k∈K} [a^r_k * (T^r_{k,c} + T_{k,p})]
```

Per-vehicle communication time (uplink):
```
T^r_{k,c} = M / q^r_k

q^r_k = B_k * log2(1 + P_k * h_k * (L^r_k)^{-β} / N_0)   (uplink rate, OFDMA)
B_k = B / Σ_k a^r_k                                         (equal bandwidth split)
L^r_k = sqrt((L_z)² + H²)                                   (distance to BS in zone z)
```

Per-vehicle compute time:
```
T_{k,p} = |D_k| * g_k / (c_k * f_k)
```

---

## 6. Optimization Problem (P1)

```
maximize over a^r:   Σ_k Φ^r(a^r_k)

subject to:
  (7a)  B / Σ_k a^r_k  ≥  B_min       (per-vehicle bandwidth lower bound)
  (7b)  a^r_k ∈ {0, 1}                 (binary selection)
  (7c)  Σ_k a^r_k = K_0                (exactly K_0 vehicles selected per round)
```

**Why P1 is intractable directly**: evaluating `p^r(a^r_k)` requires knowing which selected vehicles will stay in coverage after `E` local epochs — i.e., future positions. This information is unavailable at the time of selection.

---

## 7. MAB Formulation

Each vehicle `k ∈ {1, …, K_0}` is modeled as a bandit arm. Each round, the BS "pulls" `K_0` arms simultaneously (combinatorial MAB). The observed reward when vehicle `k` is selected in round `r` is `Φ^r(a^r_k)` (the realized utility after the round completes).

**Exploitation**: select vehicles with high historical utility (they tend to stay in range and contribute good updates)
**Exploration**: select vehicles not recently chosen (to discover vehicles in other zones that may be better)

---

## 8. Exploitation: Discounted Empirical Average

### Discounted Selection Count

```
M^r(λ, a^r_k) = Σ_{τ=1}^{r}  λ^{r-τ} · 1(a^τ_k = 1)
```

- `λ ∈ [0, 1]` is the discount factor
- `λ^{r-τ}` exponentially down-weights selections from τ rounds ago
- When `λ = 1`: uniform weight on all past selections (standard MAB)
- When `λ < 1`: recent selections count more — appropriate because the vehicle population changes over time as vehicles enter/exit the road

Mathematical interpretation: `M^r(λ, a^r_k)` is a geometric series. If vehicle `k` was selected in every round from 1 to `r`:
```
M^r = Σ_{τ=1}^{r} λ^{r-τ} = (1 - λ^r) / (1 - λ)
```

### Discounted Empirical Average Utility (Eq. 8)

```
Φ̄^r(λ, a^r_k) = [Σ_{τ=1}^{r}  λ^{r-τ} · 1(a^τ_k = 1) · Φ^(r)(a^τ_k)]
                  ─────────────────────────────────────────────────────────
                                    M^r(λ, a^r_k)
```

This is vehicle `k`'s estimated quality: a discount-weighted average of all past realized utilities. The numerator accumulates discounted utility; dividing by `M^r` normalizes.

**Recurrence form** (useful for implementation — avoids full history scan):
```
numerator^r_k = λ * numerator^{r-1}_k  +  1(a^r_k = 1) * Φ^r(a^r_k)
M^r_k         = λ * M^{r-1}_k          +  1(a^r_k = 1)
Φ̄^r_k         = numerator^r_k / M^r_k    (if M^r_k > 0)
```

---

## 9. Exploration: UCB Index (Eq. 9)

```
c_k(λ, a^r_k) = sqrt( 2 * log(n(r, λ)) / M^r(λ, a^r_k) )
```

Where `n(r, λ)` is the **total discounted selection count across all vehicles**:
```
n(r, λ) = Σ_k M^r(λ, a^r_k)
```

**Behavior analysis**:

| Scenario | Effect on c_k |
|---|---|
| Vehicle `k` never selected | `M^r_k = 0` → `c_k = ∞` (highest priority; see §12 for implementation) |
| Vehicle `k` rarely selected | Small `M^r_k`, large `c_k` → boosted score → gets selected soon |
| Vehicle `k` frequently selected | Large `M^r_k`, small `c_k` → exploration pressure elsewhere |
| All vehicles selected equally | UCB degenerates toward pure exploitation |

**Origin**: This is UCB1 (Auer et al. 2002) adapted for the discounted setting. The standard UCB1 index for arm `k` after `n` total pulls and `m_k` pulls of arm `k` is `sqrt(2 log(n) / m_k)`. The paper replaces raw counts with their discounted equivalents.

---

## 10. Combined UCB Score (Eq. 10)

```
U_k(a^r_k) = Φ̄^r(λ, a^r_k)  +  c_k(λ, a^r_k)
```

This is the optimistic estimate of vehicle `k`'s utility: the historical average plus an exploration bonus inversely proportional to how often `k` has been selected.

**Selection rule**: At round `r`, select the `K_0` vehicles with the largest `U_k` values:
```
S^r = { k : U_k ranks in top K_0 among all vehicles }
```

The paper states: "all the selection choices are learned from past rounds of UCB scores" — meaning the UCB scores are updated each round after observing the realized utility, and the next round's selection uses the updated scores.

---

## 11. Full Algorithm (Algorithm 1)

```
Input:  vehicle locations {x^r_k}, number to select K_0
Output: selected vehicle set S^r for each round r

for r = 1 to R:
    if r == 1:
        Select K_0 vehicles uniformly at random → S^0
        Initialize: for all k, M^r_k = 0, numerator_k = 0
    else:
        For each vehicle k:
            Update M^r_k  = λ * M^{r-1}_k  +  1(k ∈ S^{r-1})
            Update numerator_k = λ * numerator_k  +  1(k ∈ S^{r-1}) * Φ^{r-1}(a^{r-1}_k)
            Compute Φ̄^r_k = numerator_k / M^r_k          (if M^r_k > 0, else Φ̄^r_k = ∞)
            Compute n(r, λ) = Σ_j M^r_j
            Compute c_k = sqrt(2 * log(n(r,λ)) / M^r_k)   (if M^r_k > 0, else c_k = ∞)
            U_k = Φ̄^r_k + c_k
        S^r = top K_0 vehicles by U_k score

    BS distributes w^r to all k ∈ S^r
    Vehicles train locally for E epochs
    Server collects uploads, computes p^r
    Server aggregates → w^{r+1}
    Server observes realized utility Φ^r for vehicles in S^r
```

---

## 12. Implementation Notes & Edge Cases

### 1. Cold start (r = 1)
No history exists. Select K_0 vehicles uniformly at random. Do not compute UCB scores. This is required because `M^r = 0` for all vehicles, making the UCB index undefined.

### 2. Never-selected vehicles (M^r_k = 0)
Division by zero in both `Φ̄^r` and `c_k`. Standard MAB convention: assign score `+∞` (infinite optimism). In practice: initialize all vehicles with `M = 0` and handle this as a special case, selecting them first before any scored vehicle.

### 3. Discount factor λ
The paper does not give an explicit value for λ. It must be tuned. Suggested range: 0.7–0.95. Lower values adapt faster to new vehicles entering the segment but forget useful history sooner. λ = 1 reduces to standard (non-discounted) UCB.

### 4. Computing n(r, λ) efficiently
`n(r, λ)` should be accumulated as a single scalar updated each round:
```
n(r, λ) = λ * n(r-1, λ) + |S^{r-1}|   (since K_0 vehicles were selected)
```
This avoids summing over all vehicles every round.

### 5. Constraint (7a) at selection time
When selecting K_0 vehicles, per-vehicle bandwidth is `B / K_0`. Constraint (7a): `B / K_0 ≥ B_min`. This implicitly bounds K_0 from above: `K_0 ≤ B / B_min`. Treat K_0 as fixed and verify it satisfies this once at configuration time.

### 6. Utility when p^r = 0
If no vehicle uploads in a round, the server keeps `w^{r+1} = w^r`. The utility for this round is:
```
Φ^r = α * 0 - (1-α) * (T_max - T_min) / (T_max - T_min) = -(1-α)
```
This negative utility is fed back into the UCB history, discouraging future selection of the same vehicle set.

### 7. Observed vs. expected utility
The utility `Φ^r` is only known *after* the round completes (p^r is observed post-hoc). The UCB score for round `r+1` uses the utility realized in round `r`. The UCB estimate at selection time is the optimistic upper bound `U_k`, not the true utility.

---

## 13. Baselines the UCB Strategy Beats

| Baseline | Description | Strategy |
|---|---|---|
| CBS (Communication-Based Selection) | Picks vehicles nearest to BS (best channel) | Greedy on current channel quality |
| RBS (Remain-Time Based Selection) | Picks vehicles with longest remaining time in coverage zone | Greedy on predicted stay time |
| Random | Picks K_0 vehicles uniformly at random | No strategy |

### Performance results (Table II)

**CIFAR-10 (ResNet-18), target 75% accuracy**:
| Method | 60 km/h (sec) | 80 km/h (sec) |
|---|---|---|
| MAB (Proposed) | 2326.27 | 2357.96 |
| CBS | 2812.62 (×1.21) | 2558.51 (×1.08) |
| RBS | 2613.61 (×1.12) | 3890.03 (×1.65) |
| Random | 2974.8 (×1.27) | N/A (never reaches 80%) |

**GTSRB (LeNet), target 90% accuracy**:
| Method | 60 km/h (sec) | 80 km/h (sec) |
|---|---|---|
| MAB (Proposed) | 320.24 | 339.66 |
| CBS | 457.67 (×1.42) | 492.72 (×1.45) |
| RBS | 398.5 (×1.24) | 384.03 (×1.13) |
| Random | 666.14 (×2.08) | 1298.26 (×3.82) |

Average improvement claimed: ~28% faster convergence.

---

## 14. Variable Index

| Symbol | Meaning |
|---|---|
| `K_0` | Number of vehicles selected per round |
| `r` | Training round index (1 to R) |
| `R` | Total number of rounds |
| `S^r` | Set of vehicles selected in round r |
| `N^r` | Set of vehicles that successfully uploaded in round r |
| `p^r` | Successful training ratio in round r: \|N^r\| / \|S^r\| |
| `1^r_k` | Dropout indicator: 1 if vehicle k stayed in coverage, 0 if it drove out |
| `a^r_k` | Binary selection indicator: 1 if vehicle k selected in round r |
| `Φ^r(a^r_k)` | Realized utility for vehicle k in round r |
| `α` | Utility weight parameter (0.6 in paper) |
| `λ` | Discount factor for MAB history (not stated explicitly in paper) |
| `M^r(λ, a^r_k)` | Discounted selection count for vehicle k before round r |
| `Φ̄^r(λ, a^r_k)` | Discounted empirical average utility for vehicle k |
| `n(r, λ)` | Total discounted selection count across all vehicles before round r |
| `c_k(λ, a^r_k)` | UCB exploration bonus for vehicle k at round r |
| `U_k(a^r_k)` | Final UCB score for vehicle k at round r |
| `η` | Local SGD learning rate (0.01 in paper) |
| `E` | Number of local SGD epochs per round |
| `T_min`, `T_max` | Min/max round durations for normalization |
| `B` | Total bandwidth (3 MHz in paper) |
| `B_min` | Minimum per-vehicle bandwidth |
| `L` | L-smoothness constant |
| `G` | Upper bound on stochastic gradient norm |
| `ε_g` | Gradient divergence bound (heterogeneity measure) |
