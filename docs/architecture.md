# System Architecture

## Overview

The system follows a Decision-Focused Learning pipeline:

Data → Deep Learning → Prediction → Optimization → Decision → Simulation → Outcome

---

## Components

### 1. Data Layer

* Synthetic or historical pricing data
* Features: price, seasonality, promotions

---

### 2. Deep Learning Layer

* LSTM-based demand prediction (TFT-inspired)
* Captures non-linear demand patterns

---

### 3. Prediction Layer

* Generates demand estimates for each SKU-price pair

---

### 4. Optimization Layer

#### OR-Tools (Exact Optimization)

* Maximizes profit
* Enforces constraints:

  * Inventory
  * One price per SKU
  * Business rules

#### PyMOO (Multi-objective Optimization)

* Optimizes:

  * Profit
  * CPI deviation
* Produces Pareto-optimal solutions

---

### 5. Decision Layer

* Selects optimal pricing strategy per SKU

---

### 6. Simulator Layer

* Evaluates:

  * Profit
  * Vendor funding
  * CPI penalties

---

## Architecture Diagram

```
                ┌──────────────────────────┐
                │   Data (Synthetic/Real)  │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │ Feature Engineering      │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │ Deep Learning Model      │
                │ (LSTM / TFT-ready)       │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │ Demand Predictions       │
                └────────────┬─────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
 ┌──────────────────────┐        ┌──────────────────────┐
 │ OR-Tools             │        │ PyMOO (NSGA-II)      │
 │ (Exact Optimization) │        │ (Multi-objective)    │
 └────────────┬─────────┘        └────────────┬─────────┘
              │                               │
              ▼                               ▼
       ┌──────────────────────────────────────────┐
       │        Pricing Decisions                 │
       └────────────┬─────────────────────────────┘
                    │
                    ▼
       ┌──────────────────────────┐
       │ Simulator                │
       └────────────┬─────────────┘
                    │
                    ▼
       ┌──────────────────────────┐
       │ Business Outcome         │
       │ (Profit)                │
       └──────────────────────────┘
```
