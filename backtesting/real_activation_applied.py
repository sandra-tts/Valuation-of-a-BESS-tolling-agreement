import pandas as pd
import numpy as np
import random
from matplotlib.ticker import MaxNLocator
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Función para calcular costes de desequilibrio y profit corregido
def compute_imbalance_costs(df_in, file, out_csv_suffix, df_indicators, n_steps_per_sim=96):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import MaxNLocator

    dfc = df_in.copy()

    # precios desequilibrio (ejemplo simple)
    dfc['price_imbalance_up_soc'] = df_indicators['686']
    dfc['price_imbalance_down_soc'] = df_indicators['687']

    dfc['price_esec_up'] = df_indicators['682']
    dfc['price_esec_down'] = df_indicators['683']

    # costes por fila
    dfc['cost_imbalance_up_soc'] = abs(dfc['soc_imbalance_up_MW']) * dfc['price_imbalance_up_soc'] * DT_H
    dfc['cost_imbalance_down_soc'] = abs(dfc['soc_imbalance_down_MW']) * dfc['price_imbalance_down_soc'] * DT_H
    dfc['cost_imbalance_up_esec'] = abs(dfc['esec_imbalance_up_MW']) * dfc['price_esec_up'] * DT_H*1.5
    dfc['cost_imbalance_down_esec'] = abs(dfc['esec_imbalance_down_MW']) * dfc['price_esec_down'] * DT_H*1.5
    dfc['cost_esec_up'] = dfc['new_banda_up_activated_MW'] * dfc['price_esec_up'] * DT_H
    dfc['cost_esec_down'] = dfc['new_banda_down_activated_MW'] * dfc['price_esec_down'] * DT_H
    # profit corregido por fila
    dfc['profit_corrected'] = (dfc['spot_revenue_EUR'] 
        + dfc['banda_revenue_EUR']
        + dfc['cost_esec_up']
        - dfc['cost_esec_down']
        + dfc['cost_imbalance_up_soc']
        - dfc['cost_imbalance_down_soc']
        - dfc['cost_imbalance_up_esec']
        - dfc['cost_imbalance_down_esec']
    )
    name_file = os.path.basename(file)

    first_part_name = name_file.split("_activated")[0]
    dfc.to_csv(f"data/backtesting_results/activation/real_activation/profit_corrected_with_spot/with_activation/corrected_{first_part_name}.csv", index=False)
    # profit total por simulación
    dfc['datetime'] = pd.to_datetime(dfc['datetime'], utc=True)

    profit_by_sim = (
        dfc.groupby(dfc['datetime'].dt.date)['profit_corrected']
        .sum()
        .reset_index()
        .rename(columns={'profit_corrected': 'total_profit_EUR'})
    )

    profit_by_sim_2 = (
        dfc.groupby(dfc['datetime'].dt.date)['net_profit_EUR']
        .sum()
        .reset_index()
        .rename(columns={'net_profit_EUR': 'original_total_profit_EUR'})
    )

    # profit medio
    mean_profit = profit_by_sim_2['original_total_profit_EUR'].mean()

    # --- Histograma de distribución de profits ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        profit_by_sim['total_profit_EUR'],
        bins=15,
        color='skyblue',
        edgecolor='black',
        alpha=0.8
    )

    # Línea vertical con el profit medio
    ax.axvline(
        x=mean_profit,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Profit original = {mean_profit:,.0f} €'
    )

    # Ejes y etiquetas
    ax.set_xlabel('Profit total por día (€)')
    ax.set_ylabel('Frecuencia (nº días)')
    ax.set_title(f'Distribución de profits totales - {out_csv_suffix}')
    ax.grid(alpha=0.3)
    ax.legend()

    # Forzar eje Y con valores enteros
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    #plt.savefig(f"data/simulacions/profit_hist_{out_csv_suffix}_3.png", dpi=150)
    plt.show()

    return profit_by_sim, mean_profit


df_indicators = pd.read_csv('data/backtesting_results/activation/real_activation/activation_real_values_enriched_2.csv')

BASE_DIR = Path(__file__).resolve().parent.parent  
ACTIVATION_DIR = BASE_DIR / "data" / "backtesting_results" / "activation"

for file in ACTIVATION_DIR.glob("*expost_optimization_horizon_2_days_SRS*.csv"):
    CAPACITY = 3.8  # MWh
    DT_H = 0.25  # horas
    ETA_CHARGE = 0.90
    ETA_DISCHARGE = 0.90
    df2 = pd.read_csv(file)

    random_value = np.random.randint(0, 2, size=len(df2))

    df2['activation_request_up'] = df_indicators['activation_up']*random_value*1.2
    df2['activation_request_down'] = df_indicators['activation_down']*(1-random_value)*1.2

    # Añadir round si consideramos que la banda activada tiene que ser discreta
    df2['new_banda_up_activated_MW'] = (df2['banda_up_offered_MW'] * df2['activation_request_up'])
    df2['new_banda_down_activated_MW'] = (df2['banda_down_offered_MW'] * df2['activation_request_down'])

    # Métode 2
    new_soc_list_2 = []
    imbalance_down_soc_list_2 = []
    imbalance_up_soc_list_2 = []
    imbalance_down_esec_list_2 = []
    imbalance_up_esec_list_2 = []

    new_soc_total = 0.0
    for i, row in df2.iterrows():
        #prev_soc = row['soc_start_MWh']
        imbalance_up, imbalance_down = 0.0, 0.0
        soc_imbalance_up, soc_imbalance_down, esec_imbalance_up, esec_imbalance_down = 0.0, 0.0, 0.0, 0.0

        charge_net = row['charge_base_MW'] + row['new_banda_down_activated_MW']
        discharge_net = row['discharge_base_MW'] + row['new_banda_up_activated_MW']

        new_soc_row = new_soc_total

        new_soc_total = new_soc_row \
            + charge_net * ETA_CHARGE * DT_H \
            - discharge_net / ETA_DISCHARGE * DT_H
        
        if new_soc_total < 0:
            imbalance_down = -new_soc_total
            new_soc_total = 0
        elif new_soc_total > CAPACITY:
            imbalance_up = new_soc_total - CAPACITY
            new_soc_total = CAPACITY

        if (imbalance_up > 0) or (imbalance_down > 0):
            
            soc_charge_discharge = new_soc_row + row['charge_base_MW']* ETA_CHARGE * DT_H - row['discharge_base_MW'] / ETA_DISCHARGE * DT_H
            if soc_charge_discharge < 0:
                soc_imbalance_down = -soc_charge_discharge
            elif soc_charge_discharge > CAPACITY:
                soc_imbalance_up = soc_charge_discharge - CAPACITY

            esec_imbalance_up = imbalance_up - soc_imbalance_up
            esec_imbalance_down = imbalance_down - soc_imbalance_down
            if esec_imbalance_down < 0 and soc_imbalance_down > 0:
                esec_imbalance_down = 0.0 
            if esec_imbalance_up < 0 and soc_imbalance_up > 0:
                esec_imbalance_up = 0.0
        new_soc_list_2.append(new_soc_row)
        imbalance_down_soc_list_2.append(soc_imbalance_down)
        imbalance_up_soc_list_2.append(soc_imbalance_up)
        imbalance_down_esec_list_2.append(esec_imbalance_down)
        imbalance_up_esec_list_2.append(esec_imbalance_up)

        if i % 100 == 0:
            print(f"i={i}, new_soc={new_soc_row}, imbalance_up={imbalance_up}, imbalance_down={imbalance_down}")

    output_file = file.with_name(
        file.stem + "_real_activation_applied.csv"
    )
    df2['new_soc_MWh'] = new_soc_list_2
    df2['soc_imbalance_down_MW'] = imbalance_down_soc_list_2
    df2['soc_imbalance_up_MW'] = imbalance_up_soc_list_2
    df2['esec_imbalance_down_MW'] = imbalance_down_esec_list_2
    df2['esec_imbalance_up_MW'] = imbalance_up_esec_list_2

    df2.to_csv('data/backtesting_results/activation/real_activation/' + str(output_file.name), index=False)
        # aplicar a ambos dataframes
    profit_df2, mean_profit_df2 = compute_imbalance_costs(df2, file, "Real activation applied", df_indicators)
pass




