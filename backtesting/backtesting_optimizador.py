import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import random
from backtesting_utils import battery_optimization_baseline_1H, battery_optimization_SRS_activation_1H, build_dataframe, compare_optimizations, update_revenue_from_predicted_decisions
from price_predictions import generate_price_predictions, generate_model_parameters
import pickle
import time

times = {}
horizon_days = [2**i for i in range(6)] 
horizon_days = [32]
model_params = generate_model_parameters()

split_date = '2025-08-31'
end_date ='2025-11-02'  # Se tiene que poner un día menos (hacemos predicción a día D+1)
models = ['baseline', 'SRS']
beta_params = {'alpha_up': 0.5107910259715398, 'beta_up': 4.350513528000816, 'alpha_down': 0.4360813177035952, 'beta_down': 3.3875718368695433}

for model_type in models:
    if model_type == 'baseline':
        optimize_battery = battery_optimization_baseline_1H
    elif model_type == 'SRS':
        optimize_battery = battery_optimization_SRS_activation_1H
    else:
        raise ValueError(f"Modelo de optimización desconocido: {model_type}")
    
    df_revenue_comparison = pd.DataFrame()
    for horizon in horizon_days:
        df_final_exante, df_final_perfect_foresight, df_final_expost, df_pred_real_errors = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        previous_soc_exante, previous_soc_expost = 0.0, 0.0
        dict_test_vs_pred = {}

        for i, day in enumerate(pd.date_range(start=split_date, end=end_date)):
            dict_optimization_input, dict_price_pred_real, dict_test_vs_pred = generate_price_predictions(model_type, horizon, day, model_params, dict_test_vs_pred)
            
            if horizon == 32 and model_type == 'SRS': 
                df_optimization_real_input, df_error = build_dataframe(model_type, dict_price_pred_real)
                df_pred_real_errors = pd.concat([df_pred_real_errors, df_error], ignore_index=True)
                
            else:
                df_optimization_real_input, _ = build_dataframe(model_type, dict_price_pred_real)
            
            df_optimization_pred_input, _ = build_dataframe(model_type, dict_optimization_input)
            df_opt_day_perfect_foresight, previous_soc_expost = optimize_battery(df_optimization_real_input, previous_soc_expost, beta_params=beta_params)
            df_opt_day_exante, previous_soc_exante = optimize_battery(df_optimization_pred_input, previous_soc_exante, beta_params=beta_params)
            df_opt_day_expost = update_revenue_from_predicted_decisions(df_opt_day_exante, df_optimization_real_input)
            df_final_expost = pd.concat([df_final_expost, df_opt_day_expost], ignore_index=True)
            df_final_exante = pd.concat([df_final_exante, df_opt_day_exante], ignore_index=True)
            df_final_perfect_foresight = pd.concat([df_final_perfect_foresight, df_opt_day_perfect_foresight], ignore_index=True)
            print('day')

        df_final_exante.to_csv(f'data/backtesting_results/activation/exante_optimization_horizon_{horizon}_days_{model_type}_activated.csv', index=False)
        df_final_expost.to_csv(f'data/backtesting_results/activation/expost_optimization_horizon_{horizon}_days_{model_type}_activated.csv', index=False)
        df_final_perfect_foresight.to_csv(f'data/backtesting_results/activation/perfect_foresight_optimization_horizon_{horizon}_days_{model_type}_activated.csv', index=False)
pass
