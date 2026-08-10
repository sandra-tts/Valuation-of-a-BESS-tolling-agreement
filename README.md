# Short-Term Valuation of Battery Energy Storage Systems (BESS) in the Spanish Electricity Markets

> Optimization and forecasting framework for the short-term valuation and operation of Battery Energy Storage Systems (BESS) participating in the Spanish electricity market.

---

## Overview

This project develops a short-term valuation framework for Battery Energy Storage Systems (BESS) operating in the Spanish electricity market.

The framework combines:

- A **Mixed-Integer Linear Programming (MILP)** model for optimal battery scheduling.
- **Machine-learning-based electricity price forecasting** using XGBoost.
- **Rolling-horizon optimization** to reproduce realistic sequential decision-making.
- Participation in both the **day-ahead energy market** and the **automatic Frequency Restoration Reserve (aFRR)** market.
- Explicit modelling of **aFRR activation** and its impact on battery operation, revenues and state of charge.
- An **ex-ante / ex-post backtesting framework** to quantify the impact of price forecast errors.

The main objective is to evaluate how different market participation strategies and optimization horizons affect BESS revenues under realistic Spanish market conditions.

The work focuses particularly on the recent reform of the Spanish secondary regulation market, which introduced separate upward and downward aFRR products. The framework therefore allows the battery to optimize both energy arbitrage and ancillary-service capacity simultaneously. :contentReference[oaicite:0]{index=0}

---
## Requirements

The project requires Python 3.x and the Python packages listed in
`requirements.txt`.

The optimization models are formulated using Pyomo and solved with
Gurobi.

> **Important:** Gurobi is not installed through `requirements.txt`.
> A separate Gurobi installation and a valid license are required to
> run the optimization models.

---

## Key Results

The main conclusions of the project are:

- **aFRR participation substantially increases BESS revenues** compared with energy-only arbitrage.
- For the studied Spanish market conditions, **1–2 day optimization horizons provide the best trade-off** between revenue and computational cost.
- Extending the optimization horizon beyond two days does **not provide systematic revenue improvements**, while significantly increasing computational effort.
- **Price forecast accuracy deteriorates with the forecasting horizon**, particularly for the spot market.
- Explicitly modelling **aFRR activation** is important because activation affects both the battery's state of charge and its effective cycling.
- Although the daily economic impact of activation may be relatively small, these differences can accumulate over long operating periods and become relevant for BESS valuation and degradation assessment.
- Daily BESS revenues cannot generally be attributed to a single market variable, since they depend on the interaction between price spreads, forecast accuracy, cycling, solar capture prices and ancillary-service revenues.
---
<img width="2400" height="1950" alt="battery_operation_horizon_2_SRS_cycled_page-0001" src="https://github.com/user-attachments/assets/ac7343d9-6e00-4424-8ef4-594ad622bef4" />
<img width="2700" height="1020" alt="daily_profit_comparison_activation_boxplot_page-0001" src="https://github.com/user-attachments/assets/15ab8bbb-21a3-4998-9be8-6137fe0e0ec9" />
<img width="2700" height="1800" alt="cumulative_revenue_differences_page-0001" src="https://github.com/user-attachments/assets/6c28ec78-0223-4466-8e66-04a2d65e75d1" />
<img width="2100" height="750" alt="EFC_before_after_activation_page-0001" src="https://github.com/user-attachments/assets/2ca350f7-2b33-43a8-9ae4-e38ae52c1290" />


## Project Architecture

The framework can be summarized as:

```text
Historical Spanish Electricity Market Data
                │
                ▼
        Data preprocessing
                │
                ▼
      Feature engineering
                │
                ▼
       XGBoost forecasting
                │
       ┌────────┴────────┐
       │                 │
       ▼                 ▼
 Forecast prices     Real prices
       │                 │
       ▼                 ▼
   Ex-ante MILP      Ex-post MILP
       │                 │
       └────────┬────────┘
                ▼
       Rolling-horizon BESS
          optimization
                │
                ▼
      Battery operation
                │
       ┌────────┴─────────┐
       ▼                  ▼
 Energy arbitrage      aFRR capacity
                            │
                            ▼
                     aFRR activation
                            │
                            ▼
                 Real SoC & imbalances
                            │
                            ▼
                 Revenue evaluation
