# Decision-Focused Learning for Retail Pricing Optimization

🚀 **Key Result:**
Achieved ~150% improvement in profit by moving from Predict-Then-Optimize (PTO) to a Decision-Focused Learning (DFL) system.

---

## 📌 Problem

Retail pricing decisions must balance:

* Demand sensitivity
* Vendor funding
* Competitive pricing (CPI)
* Inventory constraints

Traditional systems optimize prediction accuracy, not business outcomes.

---

## 🎯 Objective

Build a system that:

* Predicts demand using Deep Learning
* Optimizes pricing using constraints
* Maximizes total profit

---

## 🧠 Solution Overview

Deep Learning → Demand Prediction → Optimization → Pricing Decision

---

## ⚙️ Key Components

### 1. Deep Learning Model

* LSTM-based (TFT-inspired)
* Captures non-linear demand behavior

---

### 2. Optimization

#### OR-Tools

* Exact optimization
* Produces best pricing decision

#### PyMOO (NSGA-II)

* Multi-objective optimization
* Explores trade-offs between:

  * Profit
  * CPI

---

### 3. Simulator

* Evaluates real-world impact:

  * Vendor funding
  * CPI penalties
  * Demand elasticity

---

## 💰 Profit Function

Profit = (Price - Cost) × Demand + Vendor - CPI Penalty

---

## 📊 Results

| Approach | Profit |
| -------- | ------ |
| PTO      | 4700   |
| DFL      | 11850  |
| Gain     | +150%  |

---

## 🚀 Run

```bash
pip install -r requirements.txt
python src/main.py
```

---

## 🔮 Future Work

* Full Temporal Fusion Transformer (TFT)
* Multi-period optimization
* Reinforcement learning pricing
* Real-time deployment APIs

---

## 👨‍💻 Author

Veeresh T.
