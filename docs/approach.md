# Approach

## Step 1: Demand Modeling

We train a model to learn:

```text
(price, season, promo) → demand
```

---

## Step 2: Optimization

We solve:

```text
Maximize Profit
```

Subject to:

* Inventory constraint
* Vendor rules
* CPI penalties

---

## Step 3: Decision-Focused Learning

Instead of optimizing prediction accuracy:

We optimize:

```text
Final Profit
```

---

## Why OR-Tools?

* Efficient combinatorial optimization
* Handles constraints naturally
* Scales to large SKU sets
