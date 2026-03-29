# 📊 Retail Pricing Optimization using Decision-Focused Learning (DFL)

---

## 🚀 Overview

Retail pricing decisions are traditionally made using a **two-stage approach**:

1. Predict demand using Machine Learning
2. Optimize prices using rule-based or mathematical optimization

However, this approach fails in real-world retail scenarios involving:

* Vendor funding (promotional allowances)
* Competitive pricing constraints (CPI)
* Promotions and markdown strategies
* Multi-objective trade-offs

---

## ❌ Problem with Traditional Approach

### 1. Predict-Then-Optimize Failure

Traditional systems optimize:

```text
maximize revenue = price × predicted_demand
```

But ignore:

* Vendor allowances (conditional funding)
* Competitive constraints (CPI)
* Promotional interactions

👉 Result:
**Accurate demand ≠ Profitable pricing decisions**

---

### 2. Disconnect Between Forecast & Profit

Even if demand prediction is accurate:

* Model predicts demand at price = ₹20
* Optimizer sets price = ₹30
* Demand changes → prediction becomes invalid

👉 This leads to **suboptimal or infeasible solutions**

---

### 3. Vendor Allowance Complexity

Vendor funding depends on:

* Promotions
* Demand thresholds
* Product type (private label vs vendor)

👉 These are **non-linear and conditional relationships**
👉 Cannot be captured in a simple two-stage pipeline

---

## ✅ Our Solution: Decision-Focused Learning (DFL)

We implement a **fully end-to-end differentiable system**:

```text
Features
   ↓
TFT Demand Model
   ↓
Differentiable Pricing Layer
   ↓
CPI Constraint Enforcement
   ↓
Demand Re-estimation (elasticity + margin)
   ↓
Vendor Allowance Computation
   ↓
Profit
   ↓
Backpropagation (End-to-End)
```

---

## 🎯 Business Objectives Solved

### 1. Gross Profit Maximization

We optimize:

```text
Profit = Revenue - Cost + Vendor Allowance - Markdown
```

Where:

* Revenue = price × demand
* Vendor Allowance = conditional funding
* Demand = function of price & promotions

---

### 2. Unified Price Intelligence

* Private labels → no vendor allowance
* Vendor products → allowance applied conditionally

Handled via:

```python
allowance = 0.2 * price * demand if promo & threshold met
```

---

### 3. CPI Compliance (Critical Constraint)

We enforce:

```text
1.00 ≤ CPI ≤ 1.07
```

Where:

```text
CPI = Σ(wᵢ × priceᵢ) / Σ(wᵢ × competitor_priceᵢ)
```

✅ Implemented using **iterative projection (hard constraint)**
(Not penalty-based)

---

### 4. Promotional Coordination

Promotions are modeled as:

```python
promos = sigmoid(threshold - demand)
```

This ensures:

* Promotions activate when demand is low
* Vendor funding is triggered correctly

---

### 5. Multi-Objective Optimization

We jointly optimize:

* Profit (maximize)
* CPI deviation (minimize)
* Vendor funding (maximize indirectly)

---

## 🧠 Deep Learning Component (TFT)

We use a simplified **Temporal Fusion Transformer (TFT)**:

* Handles feature interactions
* Models demand non-linearly
* Supports temporal inputs

### Model Flow:

```text
Input Features → Linear → LSTM → Attention → Output (Demand)
```

Output:

```text
Demand = f(features, price, promo)
```

---

## ⚙️ Optimization using PyMOO

We use **NSGA-II (multi-objective genetic algorithm)**.

---

### 1. Problem Formulation

Each solution vector:

```text
X = [price_index_1 ... price_index_n, promo_1 ... promo_n]
```

* Price index → selects from discrete price options
* Promo → binary (0 or 1)

---

### 2. Objective Functions

We define:

```text
F1 = -Profit   (maximize profit)
F2 = |CPI - 1.03|  (target CPI center)
```

---

### 3. Constraint Functions

#### Linear Constraints (CPI bounds):

```text
G1 = 1.00 - CPI ≤ 0
G2 = CPI - 1.07 ≤ 0
```

---

#### Non-Linear Constraints:

* Demand depends on:

  * price (exponential elasticity)
  * margin (sigmoid)
* Vendor allowance depends on:

  * promo AND threshold AND demand

---

### 4. Feasible Space

* Prices ∈ discrete set (e.g., 5 options per SKU)
* Promos ∈ {0, 1}

For **N SKUs**:

```text
Total combinations = (price_options × promo_options)^N
                   = (5 × 2)^N = 10^N
```

For N = 20:

```text
10^20 possible combinations (intractable)
```

👉 Hence we use **evolutionary optimization (NSGA-II)**

---

## 🔥 Decision-Focused Learning (DFL)

### Key Idea:

Instead of:

```text
Train demand model → Optimize later
```

We do:

```text
Train demand model THROUGH optimization objective
```

---

### Loss Function:

```text
Loss = -Profit
```

Where:

* Profit depends on:

  * Demand (from model)
  * Prices (from optimization layer)

👉 This creates a **closed learning loop**

---

## 🔒 CPI Enforcement (Critical Innovation)

Instead of penalty:

```text
Loss += penalty(CPI violation)
```

We use **iterative projection**:

```python
for _ in range(k):
    cpi = compute_cpi(prices)
    prices = prices * (target_cpi / cpi)
```

✅ Guarantees feasibility
✅ Stable training
✅ No hyperparameter tuning

---

## 📊 Final Results

* Profit increases consistently across epochs
* CPI remains stable at ~1.03
* Prices converge to realistic values
* Promotions activate meaningfully

---

## 🏁 Conclusion

We successfully transformed:

❌ Predict → Optimize (Disconnected)
➡️
✅ **End-to-End Decision-Focused Learning System**

---

## 💡 Key Advantages

* Handles vendor funding complexity
* Enforces competitive constraints structurally
* Learns pricing decisions directly from profit
* Scales to combinatorial optimization problems

---

## 🚀 Future Improvements

* Learn promotion policy (instead of heuristic)
* Add vendor tiered funding
* Model competitor reaction (game theory)
* Introduce cross-SKU demand cannibalization

---

## 📌 Summary

This solution delivers a **production-grade retail pricing system** that:

✔ Maximizes profit
✔ Maintains CPI compliance
✔ Integrates vendor funding
✔ Uses end-to-end learning

👉 Fully aligned with real-world retail pricing challenges.
