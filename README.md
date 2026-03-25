# Decision-Focused Learning for Retail Pricing Optimization

🚀 Key Result:
Achieved ~150% improvement in profit by shifting from Predict-Then-Optimize (PTO) to Decision-Focused Learning (DFL) approach.

## 📌 Problem Statement

Retail pricing decisions are complex due to:

* Demand sensitivity to price
* Vendor promotional funding
* Competitive Price Index (CPI) constraints
* Inventory limitations

Traditional approaches:

> Predict demand → Apply rules

❌ Fail to optimize **business outcomes (profit)**

---

## 🎯 Objective

Build a **Decision-Focused Learning (DFL)** system that:

* Predicts demand using ML
* Optimizes pricing using constraints
* Maximizes **total profit**, not just prediction accuracy

---

## 🧠 Solution Overview

We built an end-to-end system:

```text
ML Model → Demand Prediction → Optimization Solver → Pricing Decision
```

---

## ⚙️ Key Components

### 1. Demand Prediction

* Model: RandomForest (proxy for TFT)
* Features:

  * Price
  * Seasonality
  * Promotion flag

---

### 2. Optimization (OR-Tools)

* Objective: Maximize profit
* Constraints:

  * Inventory
  * CPI (soft penalty)
  * Vendor funding rules

---

### 3. Multi-SKU Optimization

* Joint decision across multiple SKUs
* Real-world scalable design

---

## 💰 Profit Function

```text
Profit = (Price - Cost) × Demand + Vendor - CPI Penalty
```

---

## 🔥 Key Insights

* Highest price ≠ highest profit
* Lower price → higher demand + vendor funding
* Optimization finds best trade-off

---

## 📊 Results

| Approach    | Profit |
| ----------- | ------ |
| PTO         | 4700   |
| DFL         | 11850  |
| Improvement | +7150  |

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/main.py
```

---

## 🧠 Future Enhancements

* Temporal Fusion Transformer (TFT)
* Reinforcement Learning
* Multi-period pricing
* Real-time pricing APIs

---

## 👨‍💻 Author

Veeresh T.
