# Approach

## 1. Problem Formulation

We formulate retail pricing as a **Decision-Focused Optimization problem**, where the goal is to directly optimize business outcomes (profit) instead of only improving prediction accuracy.

---

## 2. Objective Function

Maximize total profit:

Profit = Σ [(Priceᵢ - Costᵢ) × Demandᵢ(Priceᵢ) + Vendorᵢ - Penaltyᵢ]

Where:

* Demand is predicted using a deep learning model
* Vendor incentives depend on pricing and demand
* CPI penalties enforce competitive pricing

---

## 3. Constraints

### Linear Constraints

* One price per SKU:
  Σ xᵢ = 1

* Inventory constraint:
  Σ Demandᵢ ≤ Inventoryᵢ

---

### Non-Linear Components

* Demand is a function of price:
  Demandᵢ = f(Priceᵢ) (Deep Learning model)

* Profit depends on demand:
  Profit = (Price - Cost) × Demand(Price)

These introduce non-linearity into the system.

---

## 4. Feasible Space

The feasible space consists of all valid pricing combinations satisfying:

* One price per SKU
* Inventory constraints
* Competitive pricing rules

For:

* N SKUs
* K price points

Total combinations:

Kⁿ

---

## 5. Combinatorial Complexity

For this implementation:

* SKUs = 3
* Price points = 19

Total combinations:

19³ = 6859

This demonstrates the need for optimization techniques instead of brute-force search.

---

## 6. Optimization Framework

### A. OR-Tools (Exact Optimization)

* Finds the **optimal price per SKU**
* Handles constraints strictly
* Used for **production-ready decision making**

---

### B. PyMOO (Multi-Objective Optimization)

We formulate a multi-objective problem:

Objectives:

1. Maximize Profit
2. Minimize CPI Deviation

* Uses NSGA-II algorithm
* Generates **Pareto-optimal solutions**
* Helps explore trade-offs between profitability and competitiveness

---

## 7. Deep Learning Model (TFT-Inspired)

We implemented a deep learning model using LSTM as a proxy for Temporal Fusion Transformers (TFT).

Why Deep Learning:

* Captures non-linear demand behavior
* Models seasonality and promotions
* Supports temporal learning

Pipeline:

Deep Learning Model → Demand Prediction → Optimization

This architecture can be extended to full TFT in production.

---

## 8. Simulator-Driven Approach

A simulator is used to evaluate:

* Price elasticity impact
* Vendor funding effects
* CPI penalty trade-offs

This enables testing multiple scenarios before deployment.

---

## 9. Summary

This system integrates:

* Deep Learning (Demand Prediction)
* Exact Optimization (OR-Tools)
* Multi-objective Optimization (PyMOO)
* Simulation (Business Evaluation)

to directly optimize business outcomes.
