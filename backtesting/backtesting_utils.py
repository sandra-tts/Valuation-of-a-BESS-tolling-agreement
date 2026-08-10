import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pathlib import Path
import pandas as pd
import numpy as np
import random
from scipy.stats import beta


def build_dataframe(model_type, input_dict: dict) -> pd.DataFrame:
    """
    Convierte un diccionario {nombre: df} en un DataFrame ancho.
    Cada df debe tener columnas 'datetime', 'pred' y opcionalmente 'real'.
    """
    df_final = None

    for name, df in input_dict.items():

        # Renombrar TODAS las columnas excepto datetime
        df_tmp = df.rename(columns={
            col: f"{name}_{col}" 
            for col in df.columns 
            if col != "datetime"
        })

        if df_final is None:
            df_final = df_tmp.copy()
        else:
            df_final = df_final.merge(df_tmp, on="datetime", how="outer")

    df_final = df_final.sort_values("datetime").reset_index(drop=True)

    if any('real' in col for col in df_final.columns):
        df_final.drop(columns=[col for col in df_final.columns if col.endswith('_pred')], inplace=True) 
        df_final = df_final.rename(columns={col: col.replace('_real', '') for col in df_final.columns if col.endswith('_real')})
    else:
        df_final.rename(columns={col: col.replace('_pred', '') for col in df_final.columns if col.endswith('_pred')}, inplace=True)

    df_error = df_final[['datetime'] + [col for col in df_final.columns if col.endswith('abs_error')] + \
                        [col for col in df_final.columns if col.endswith('days_since_start')] + [col for col in df_final.columns if col.endswith('percentage_error')]].copy()    
    df_final.drop(columns=[col for col in df_final.columns if col.endswith('abs_error') or col.endswith('days_since_start') or col.endswith('percentage_error')], errors='ignore', inplace=True)

    return df_final, df_error

def battery_optimization_SRS(df_pred_window, previous_soc, POWER_MAX_MW=1.0, CAPACITY=3.8, ETA_CHARGE=0.9, ETA_DISCHARGE=0.9, DT_H=0.25, final_soc=0.0):
    BIG_M=1e5
    df= df_pred_window.copy()

    # ----------------
    # Inicializar parámetros desde dataframe
    # ----------------
    def make_param(col):
        return {t: float(v) for t, v in zip(df['datetime'], df[col])}

    PRICE_SPOT      = make_param('SPOT')          # [€/MWh]
    PRICE_BANDA_UP  = make_param('BANDA_subir')   # [€/MW] (por QH — no multiplicar por DT_H)
    PRICE_BANDA_DN  = make_param('BANDA_bajar')   # [€/MW] (por QH)

    # ----------------
    # Crear modelo
    # ----------------
    model = pyo.ConcreteModel()

    # Set de tiempos (por fechas)
    model.T = pyo.Set(initialize=df['datetime'].tolist(), ordered=True)
    times = list(model.T)

    # ----------------
    # Parámetros del modelo
    # ----------------
    model.price_spot = pyo.Param(model.T, initialize=PRICE_SPOT)
    model.price_banda_up = pyo.Param(model.T, initialize=PRICE_BANDA_UP)
    model.price_banda_down = pyo.Param(model.T, initialize=PRICE_BANDA_DN)

    # activation_request 
    ACT_REQ_UP = {}
    ACT_REQ_DN = {}
    for t in times:
        r = 0  # número aleatorio entre 0 y 1
        choice = random.choice(["up", "down"])
        if choice == "up":
            ACT_REQ_UP[t] = r
            ACT_REQ_DN[t] = 0
        elif choice == "down":
            ACT_REQ_UP[t] = 0
            ACT_REQ_DN[t] = r
        # else:  # "none"
        #     ACT_REQ_UP[(ti, sim)] = 0.0
        #     ACT_REQ_DN[(ti, sim)] = 0.0

    model.activation_request_up = pyo.Param(model.T, initialize=ACT_REQ_UP, mutable=True, within=pyo.NonNegativeReals)
    model.activation_request_down = pyo.Param(model.T, initialize=ACT_REQ_DN, mutable=True, within=pyo.NonNegativeReals)

    # ----------------
    # Variables
    # ----------------
    model.charge_base_MW = pyo.Var(model.T, bounds=(0, POWER_MAX_MW))
    model.discharge_base_MW = pyo.Var(model.T, bounds=(0, POWER_MAX_MW))
    model.uc = pyo.Var(model.T, domain=pyo.Binary)
    model.ud = pyo.Var(model.T, domain=pyo.Binary)

    model.banda_offered_up = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 2 * POWER_MAX_MW))
    model.banda_offered_down = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 2 * POWER_MAX_MW))

    model.soc_MWh = pyo.Var(model.T, bounds=(0, CAPACITY))
    model.banda_up_activated = pyo.Var(model.T, bounds=(0, 2 * POWER_MAX_MW))
    model.banda_down_activated = pyo.Var(model.T, bounds=(0, 2 * POWER_MAX_MW))

    # ----------------
    # Restricciones
    # ----------------

    # Exclusividad carga/descarga (binarios)
    def exclusivity_rule(m, t):
        return m.uc[t] + m.ud[t] <= 1
    model.exclusivity = pyo.Constraint(model.T, rule=exclusivity_rule)

    # Límites base condicionados a los binarios (si uc=0 entonces carga_base debe ser 0)
    def base_charge_limit_rule(m, t):
        return m.charge_base_MW[t] <= POWER_MAX_MW * m.uc[t]
    model.base_charge_limit = pyo.Constraint(model.T, rule=base_charge_limit_rule)

    def base_discharge_limit_rule(m, t):
        return m.discharge_base_MW[t] <= POWER_MAX_MW * m.ud[t]
    model.base_discharge_limit = pyo.Constraint(model.T, rule=base_discharge_limit_rule)


    # Restricciones de bandas ofertadas
    # Caso carga (uc=1)
    def banda_up_carga_soc_rule(m, t):
        return m.banda_offered_up[t] <= m.soc_MWh[t]/DT_H + m.charge_base_MW[t] + (1 - m.uc[t]) * BIG_M
    model.banda_up_carga_soc = pyo.Constraint(model.T, rule=banda_up_carga_soc_rule)

    def banda_up_carga_max_rule(m, t):
        return m.banda_offered_up[t] <= m.charge_base_MW[t] + POWER_MAX_MW + (1 - m.uc[t]) * BIG_M
    model.banda_up_carga_max = pyo.Constraint(model.T, rule=banda_up_carga_max_rule)

    # Caso descarga (ud=1)
    def banda_up_descarga_soc_rule(m, t):
        return m.banda_offered_up[t] <= m.soc_MWh[t]/DT_H + (1 - m.ud[t]) * BIG_M
    model.banda_up_descarga_soc = pyo.Constraint(model.T, rule=banda_up_descarga_soc_rule)

    def banda_up_descarga_max_rule(m, t):
        return m.banda_offered_up[t] <= POWER_MAX_MW - m.discharge_base_MW[t] + (1 - m.ud[t]) * BIG_M
    model.banda_up_descarga_max = pyo.Constraint(model.T, rule=banda_up_descarga_max_rule)


    # Caso carga (uc=1)
    def banda_down_carga_rule(m, t):
        return m.banda_offered_down[t] <= POWER_MAX_MW - m.charge_base_MW[t] + (1 - m.uc[t]) * BIG_M
    model.banda_down_carga = pyo.Constraint(model.T, rule=banda_down_carga_rule)

    def banda_down_carga_cap_rule(m, t):
        return m.banda_offered_down[t] <= (CAPACITY - m.soc_MWh[t])/DT_H + (1 - m.uc[t]) * BIG_M
    model.banda_down_carga_cap = pyo.Constraint(model.T, rule=banda_down_carga_cap_rule)

    # Caso descarga (ud=1)
    def banda_down_descarga_rule(m, t):
        return m.banda_offered_down[t] <= m.discharge_base_MW[t] + POWER_MAX_MW + (1 - m.ud[t]) * BIG_M
    model.banda_down_descarga = pyo.Constraint(model.T, rule=banda_down_descarga_rule)

    def banda_down_descarga_cap_rule(m, t):
        return m.banda_offered_down[t] <= m.discharge_base_MW[t] + (CAPACITY - m.soc_MWh[t])/DT_H + (1 - m.ud[t]) * BIG_M
    model.banda_down_descarga_cap = pyo.Constraint(model.T, rule=banda_down_descarga_cap_rule)

    # Caso idle: ni carga ni descarga (uc=0, ud=0)
    def banda_up_idle_max_rule(m, t):
        return m.banda_offered_up[t] <= POWER_MAX_MW + BIG_M * (m.uc[t] + m.ud[t])
    model.banda_up_idle_max = pyo.Constraint(model.T, rule=banda_up_idle_max_rule)

    def banda_up_idle_soc_rule(m, t):
        return m.banda_offered_up[t] <= m.soc_MWh[t]/DT_H + BIG_M * (m.uc[t] + m.ud[t])
    model.banda_up_idle_soc = pyo.Constraint(model.T, rule=banda_up_idle_soc_rule)

    def banda_down_idle_max_rule(m, t):
        return m.banda_offered_down[t] <= POWER_MAX_MW + BIG_M*(m.uc[t] + m.ud[t])
    model.banda_down_idle_max = pyo.Constraint(model.T, rule=banda_down_idle_max_rule)

    def banda_down_idle_cap_rule(m, t):
        return m.banda_offered_down[t] <= (CAPACITY - m.soc_MWh[t])/DT_H + BIG_M*(m.uc[t] + m.ud[t])
    model.banda_down_idle_cap = pyo.Constraint(model.T, rule=banda_down_idle_cap_rule)

    # Activación aceptada por la batería <= disponibilidad ofertada
    def activation_up_rule(m, t):
        return m.banda_up_activated[t] == m.activation_request_up[t] * m.banda_offered_up[t]

    model.activation_up_con = pyo.Constraint(model.T, rule=activation_up_rule)

    def activation_down_rule(m, t):
        return m.banda_down_activated[t] == m.activation_request_down[t] * m.banda_offered_down[t]

    model.activation_down_con = pyo.Constraint(model.T, rule=activation_down_rule)

    # SoC inicial y final (MW-equivalente)
    def soc_init_rule(m):
        return m.soc_MWh[times[0]] == previous_soc
    model.soc_init = pyo.Constraint(rule=soc_init_rule)

    # Dinámica del SoC (en MW-equivalente)
    def soc_dynamics_rule(m, t):
        t_list = list(m.T)
        idx = t_list.index(t)
        if idx == len(t_list) - 1:
            return pyo.Constraint.Skip
        t_next = t_list[idx + 1]
        charge_net_MW = (m.charge_base_MW[t] + m.banda_down_activated[t])
        discharge_net_MW = (m.discharge_base_MW[t] + m.banda_up_activated[t])
        return m.soc_MWh[t_next] == m.soc_MWh[t] + (charge_net_MW * ETA_CHARGE)*DT_H - (discharge_net_MW / ETA_DISCHARGE)*DT_H
    model.soc_dyn = pyo.Constraint(model.T, rule=soc_dynamics_rule)

    # ----------------
    # Objetivo: valor esperado (€/día)
    # ----------------
    def obj_rule(m):
        total = sum(
            m.price_spot[t] * (m.discharge_base_MW[t] - m.charge_base_MW[t]) * DT_H
            + m.price_banda_up[t] * m.banda_offered_up[t]
            + m.price_banda_down[t] * m.banda_offered_down[t]
            for t in m.T
        )
        return total
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # ----------------
    # Resolver
    # ----------------
    solver = SolverFactory('gurobi')
    res = solver.solve(model, tee=False)
    print("Solver termination:", res.solver.termination_condition)

    # ----------------
    # Resultados y CSV final (incluye activation_request y SoC inicio/fin)
    # ----------------
    rows = []
    for t in model.T:
        charge_base = pyo.value(model.charge_base_MW[t])
        discharge_base = pyo.value(model.discharge_base_MW[t])
        uc = pyo.value(model.uc[t])
        ud = pyo.value(model.ud[t])
        banda_up = pyo.value(model.banda_offered_up[t])
        banda_down = pyo.value(model.banda_offered_down[t])
        act_req_up = pyo.value(model.activation_request_up[t])
        act_req_down = pyo.value(model.activation_request_down[t])
        banda_up_activated = pyo.value(model.banda_up_activated[t])
        banda_down_activated = pyo.value(model.banda_down_activated[t])

        # SoC inicio (MWh)
        soc_start_MWh = pyo.value(model.soc_MWh[t])

        # Valores económicos por fila (€/QH)
        spot_term = model.price_spot[t] * (discharge_base - charge_base)*DT_H
        banda_term = model.price_banda_up[t] * banda_up + model.price_banda_down[t] * banda_down
        net = spot_term + banda_term

        rows.append({
            'datetime': t,
            'uc': abs(uc),
            'ud': abs(ud),
            'charge_base_MW': charge_base,
            'discharge_base_MW': discharge_base,
            'banda_up_offered_MW': banda_up,
            'banda_down_offered_MW': banda_down,
            'activation_request_up_MW': act_req_up,
            'activation_request_down_MW': act_req_down,
            'banda_up_activated_MW': banda_up_activated,
            'banda_down_activated_MW': banda_down_activated,
            'soc_start_MWh': soc_start_MWh,
            'spot_revenue_EUR': spot_term,
            'banda_revenue_EUR': banda_term,
            'net_profit_EUR': net
        })

    out = pd.DataFrame(rows).sort_values('datetime').reset_index(drop=True)
    df_first_day = out.iloc[:96]
    
    if len(out) > 96:
        final_soc = out['soc_start_MWh'].iloc[96]
    else:
        final_soc = out['soc_start_MWh'].iloc[95]
    print(f'Optimization completed. Final SoC: {final_soc} MWh')

    return df_first_day, final_soc

def battery_optimization_SRS_1H(df_pred_window, previous_soc, POWER_MAX_MW=1.0, CAPACITY=3.8, ETA_CHARGE=0.9, ETA_DISCHARGE=0.9, DT_H=0.25, final_soc=0.0, beta_params=None):
    BIG_M=1e5
    df= df_pred_window.copy()

    # ----------------
    # Inicializar parámetros desde dataframe
    # ----------------
    def make_param(col):
        return {t: float(v) for t, v in zip(df['datetime'], df[col])}

    PRICE_SPOT      = make_param('SPOT')          # [€/MWh]
    PRICE_BANDA_UP  = make_param('BANDA_subir')   # [€/MW] (por QH — no multiplicar por DT_H)
    PRICE_BANDA_DN  = make_param('BANDA_bajar')   # [€/MW] (por QH)

    # ----------------
    # Crear modelo
    # ----------------
    model = pyo.ConcreteModel()


    time_list = df['datetime'].tolist()        
    soc_time_list = time_list + [time_list[-1] + pd.Timedelta(hours=DT_H)]

    n_periods = len(time_list)

    ### NEW SECTION → índice entero paralelo a timestamps
    T_idx = list(range(n_periods))   # 0..N-1
    soc_idx = list(range(n_periods+1))

    # Mapping índice ↔ timestamp
    idx_to_ts = {i: time_list[i] for i in T_idx}
    ts_to_idx = {ts: i for i, ts in idx_to_ts.items()}

    soc_idx_to_ts = {i: soc_time_list[i] for i in soc_idx}
    ts_to_soc_idx = {soc_time_list[i]: i for i in soc_idx}

    model.T = pyo.Set(initialize=time_list, ordered=True)          # Para potencia y precios
    model.T_soc = pyo.Set(initialize=soc_time_list, ordered=True)  # Para SOC (97 puntos)
    model.T_idx = pyo.Set(initialize=T_idx, ordered=True)
    model.T_soc_idx = pyo.Set(initialize=soc_idx, ordered=True)
    # times = list(model.T)
    # soc_times = list(model.T_soc)
    # ----------------
    # Parámetros del modelo
    # ----------------
    model.price_spot = pyo.Param(model.T, initialize=PRICE_SPOT)
    model.price_banda_up = pyo.Param(model.T, initialize=PRICE_BANDA_UP)
    model.price_banda_down = pyo.Param(model.T, initialize=PRICE_BANDA_DN)

    p_up, p_dn = 0.8620585457979225, 0.8420396600566572
    q_up = p_up / (p_up + p_dn)
    q_dn = 1 - q_up

    # activation_request 
    ACT_REQ_UP = {}
    ACT_REQ_DN = {}
    for t in time_list:
        if np.random.rand() < q_up:
            ACT_REQ_UP[t] = 0.0
            ACT_REQ_DN[t] = 0.0
        else:
            ACT_REQ_UP[t] = 0.0
            ACT_REQ_DN[t] = 0.0

    model.activation_request_up = pyo.Param(model.T, initialize=ACT_REQ_UP, mutable=True, within=pyo.NonNegativeReals)
    model.activation_request_down = pyo.Param(model.T, initialize=ACT_REQ_DN, mutable=True, within=pyo.NonNegativeReals)

    # ----------------
    # Variables
    # ----------------
    model.charge_base_MW = pyo.Var(model.T, bounds=(0, POWER_MAX_MW))
    model.discharge_base_MW = pyo.Var(model.T, bounds=(0, POWER_MAX_MW))
    model.uc = pyo.Var(model.T, domain=pyo.Binary)
    model.ud = pyo.Var(model.T, domain=pyo.Binary)

    model.banda_offered_up = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 2 * POWER_MAX_MW))
    model.banda_offered_down = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 2 * POWER_MAX_MW))

    model.soc_MWh = pyo.Var(model.T_soc, bounds=(0, CAPACITY))
    model.delta_soc = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.banda_up_activated = pyo.Var(model.T, bounds=(0, 2 * POWER_MAX_MW))
    model.banda_down_activated = pyo.Var(model.T, bounds=(0, 2 * POWER_MAX_MW))

    # ----------------
    # Restricciones
    # ----------------

    # Exclusividad carga/descarga (binarios)
    def exclusivity_rule(m, t):
        return m.uc[t] + m.ud[t] <= 1
    model.exclusivity = pyo.Constraint(model.T, rule=exclusivity_rule)

    # Límites base condicionados a los binarios (si uc=0 entonces carga_base debe ser 0)
    def base_charge_limit_rule(m, t):
        return m.charge_base_MW[t] <= POWER_MAX_MW * m.uc[t]
    model.base_charge_limit = pyo.Constraint(model.T, rule=base_charge_limit_rule)

    def base_discharge_limit_rule(m, t):
        return m.discharge_base_MW[t] <= POWER_MAX_MW * m.ud[t]
    model.base_discharge_limit = pyo.Constraint(model.T, rule=base_discharge_limit_rule)


    # Restricciones de bandas ofertadas
    # Caso carga (uc=1)
    def banda_up_carga_soc_rule(m, t):
        return m.banda_offered_up[t] <= m.soc_MWh[t]/DT_H + m.charge_base_MW[t] + (1 - m.uc[t]) * BIG_M
    model.banda_up_carga_soc = pyo.Constraint(model.T, rule=banda_up_carga_soc_rule)

    def banda_up_carga_max_rule(m, t):
        return m.banda_offered_up[t] <= m.charge_base_MW[t] + POWER_MAX_MW + (1 - m.uc[t]) * BIG_M
    model.banda_up_carga_max = pyo.Constraint(model.T, rule=banda_up_carga_max_rule)

    # Caso descarga (ud=1)
    def banda_up_descarga_soc_rule(m, t):
        return m.banda_offered_up[t] <= m.soc_MWh[t]/DT_H + (1 - m.ud[t]) * BIG_M
    model.banda_up_descarga_soc = pyo.Constraint(model.T, rule=banda_up_descarga_soc_rule)

    def banda_up_descarga_max_rule(m, t):
        return m.banda_offered_up[t] <= POWER_MAX_MW - m.discharge_base_MW[t] + (1 - m.ud[t]) * BIG_M
    model.banda_up_descarga_max = pyo.Constraint(model.T, rule=banda_up_descarga_max_rule)


    # Caso carga (uc=1)
    def banda_down_carga_rule(m, t):
        return m.banda_offered_down[t] <= POWER_MAX_MW - m.charge_base_MW[t] + (1 - m.uc[t]) * BIG_M
    model.banda_down_carga = pyo.Constraint(model.T, rule=banda_down_carga_rule)

    def banda_down_carga_cap_rule(m, t):
        return m.banda_offered_down[t] <= (CAPACITY - m.soc_MWh[t])/DT_H + (1 - m.uc[t]) * BIG_M
    model.banda_down_carga_cap = pyo.Constraint(model.T, rule=banda_down_carga_cap_rule)

    # Caso descarga (ud=1)
    def banda_down_descarga_rule(m, t):
        return m.banda_offered_down[t] <= m.discharge_base_MW[t] + POWER_MAX_MW + (1 - m.ud[t]) * BIG_M
    model.banda_down_descarga = pyo.Constraint(model.T, rule=banda_down_descarga_rule)

    def banda_down_descarga_cap_rule(m, t):
        return m.banda_offered_down[t] <= m.discharge_base_MW[t] + (CAPACITY - m.soc_MWh[t])/DT_H + (1 - m.ud[t]) * BIG_M
    model.banda_down_descarga_cap = pyo.Constraint(model.T, rule=banda_down_descarga_cap_rule)

    # Caso idle: ni carga ni descarga (uc=0, ud=0)
    def banda_up_idle_max_rule(m, t):
        return m.banda_offered_up[t] <= POWER_MAX_MW + BIG_M * (m.uc[t] + m.ud[t])
    model.banda_up_idle_max = pyo.Constraint(model.T, rule=banda_up_idle_max_rule)

    def banda_up_idle_soc_rule(m, t):
        return m.banda_offered_up[t] <= m.soc_MWh[t]/DT_H + BIG_M * (m.uc[t] + m.ud[t])
    model.banda_up_idle_soc = pyo.Constraint(model.T, rule=banda_up_idle_soc_rule)

    def banda_down_idle_max_rule(m, t):
        return m.banda_offered_down[t] <= POWER_MAX_MW + BIG_M*(m.uc[t] + m.ud[t])
    model.banda_down_idle_max = pyo.Constraint(model.T, rule=banda_down_idle_max_rule)

    def banda_down_idle_cap_rule(m, t):
        return m.banda_offered_down[t] <= (CAPACITY - m.soc_MWh[t])/DT_H + BIG_M*(m.uc[t] + m.ud[t])
    model.banda_down_idle_cap = pyo.Constraint(model.T, rule=banda_down_idle_cap_rule)

    # Activación aceptada por la batería <= disponibilidad ofertada
    def activation_up_rule(m, t):
        return m.banda_up_activated[t] == m.activation_request_up[t] * m.banda_offered_up[t]

    model.activation_up_con = pyo.Constraint(model.T, rule=activation_up_rule)

    def activation_down_rule(m, t):
        return m.banda_down_activated[t] == m.activation_request_down[t] * m.banda_offered_down[t]

    model.activation_down_con = pyo.Constraint(model.T, rule=activation_down_rule)
    

    # SoC inicial y final (MW-equivalente)
    def soc_init_rule(m):
        return m.soc_MWh[soc_time_list[0]] == previous_soc
    model.soc_init = pyo.Constraint(rule=soc_init_rule)

    # Dinámica del SoC (en MW-equivalente)
    # def soc_dynamics_rule(m, t):
    #     times = list(m.T)        # 00:00 ... 23:45
    #     soc_times = list(m.T_soc)  # 00:00 ... 23:45, 24:00

    #     idx = times.index(t)
    #     t está en posición idx en times, y en la misma posición idx en soc_times
    #     t_soc = soc_times[idx]       # mismo instante que t
    #     t_next_soc = soc_times[idx+1] 

    #     charge_net_MW = (m.charge_base_MW[t] + m.banda_down_activated[t])
    #     discharge_net_MW = (m.discharge_base_MW[t] + m.banda_up_activated[t])
    #     return m.soc_MWh[t_next_soc] == m.soc_MWh[t_soc] + (charge_net_MW * ETA_CHARGE)*DT_H - (discharge_net_MW / ETA_DISCHARGE)*DT_H
    # model.soc_dyn = pyo.Constraint(model.T, rule=soc_dynamics_rule)

    # Dinámica del SoC
    def soc_dynamics_rule(m, t):
        idx = ts_to_idx[t]
        t_soc = soc_time_list[idx]
        t_next_soc = soc_time_list[idx+1]

        charge_net_MW = (m.charge_base_MW[t] + m.banda_down_activated[t])
        discharge_net_MW = (m.discharge_base_MW[t] + m.banda_up_activated[t])

        return m.soc_MWh[t_next_soc] == m.soc_MWh[t_soc] \
               + charge_net_MW * ETA_CHARGE * DT_H \
               - discharge_net_MW / ETA_DISCHARGE * DT_H

    model.soc_dyn = pyo.Constraint(model.T, rule=soc_dynamics_rule)

    # ABS del SOC usando índices enteros
    def abs_upper(m, i):
        if i == n_periods - 1:
            return pyo.Constraint.Skip
        t = idx_to_ts[i]
        t_next = soc_time_list[i+1]
        return m.delta_soc[i] >= m.soc_MWh[t_next] - m.soc_MWh[t]

    def abs_lower(m, i):
        if i == n_periods - 1:
            return pyo.Constraint.Skip
        t = idx_to_ts[i]
        t_next = soc_time_list[i+1]
        return m.delta_soc[i] >= -(m.soc_MWh[t_next] - m.soc_MWh[t])

    model.abs_upper = pyo.Constraint(model.T_idx, rule=abs_upper)
    model.abs_lower = pyo.Constraint(model.T_idx, rule=abs_lower)

    timesteps = 96
    # ≈≈≈ RESTRICCIÓN CICLOS ≈≈≈  
    n_days = n_periods // timesteps
    day_slices = [list(range(d*timesteps, (d+1)*timesteps)) for d in range(n_days)]

    def limit_cycles(m, d):
        return sum(m.delta_soc[i] for i in day_slices[d]) <= 4 * CAPACITY

    model.daily_cycle_limit = pyo.Constraint(range(n_days), rule=limit_cycles)

    # ----------------
    # Objetivo: valor esperado (€/día)
    # ----------------
    def obj_rule(m):
        total = sum(
            m.price_spot[t] * (m.discharge_base_MW[t] - m.charge_base_MW[t]) * DT_H
            + m.price_banda_up[t] * m.banda_offered_up[t]
            + m.price_banda_down[t] * m.banda_offered_down[t]
            
            for t in m.T
        )
        return total
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # ----------------
    # Resolver
    # ----------------
    solver = SolverFactory('gurobi')
    res = solver.solve(model, tee=False)
    print("Solver termination:", res.solver.termination_condition)

    # ----------------
    # Resultados y CSV final (incluye activation_request y SoC inicio/fin)
    # ----------------
    rows = []
    for t in model.T:
        charge_base = pyo.value(model.charge_base_MW[t])
        discharge_base = pyo.value(model.discharge_base_MW[t])
        uc = pyo.value(model.uc[t])
        ud = pyo.value(model.ud[t])
        banda_up = pyo.value(model.banda_offered_up[t])
        banda_down = pyo.value(model.banda_offered_down[t])
        act_req_up = pyo.value(model.activation_request_up[t])
        act_req_down = pyo.value(model.activation_request_down[t])
        banda_up_activated = pyo.value(model.banda_up_activated[t])
        banda_down_activated = pyo.value(model.banda_down_activated[t])

        # SoC inicio (MWh)
        soc_start_MWh = pyo.value(model.soc_MWh[t])

        # Valores económicos por fila (€/QH)
        spot_term = model.price_spot[t] * (discharge_base - charge_base)*DT_H
        banda_term = model.price_banda_up[t] * banda_up + model.price_banda_down[t] * banda_down
        net = spot_term + banda_term

        rows.append({
            'datetime': t,
            'uc': abs(uc),
            'ud': abs(ud),
            'charge_base_MW': charge_base,
            'discharge_base_MW': discharge_base,
            'banda_up_offered_MW': banda_up,
            'banda_down_offered_MW': banda_down,
            'activation_request_up_MW': act_req_up,
            'activation_request_down_MW': act_req_down,
            'banda_up_activated_MW': banda_up_activated,
            'banda_down_activated_MW': banda_down_activated,
            'soc_start_MWh': soc_start_MWh,
            'spot_revenue_EUR': spot_term,
            'banda_revenue_EUR': banda_term,
            'net_profit_EUR': net
        })

    out = pd.DataFrame(rows).sort_values('datetime').reset_index(drop=True)
    df_first_day = out.iloc[:96]
    
    if timesteps < len(soc_idx_to_ts):
        final_soc_time = soc_idx_to_ts[timesteps]   # timestamp del SOC en t=192        
    else:
        # horizonte más corto (muy raro)
        final_soc_time = soc_idx_to_ts[len(soc_idx_to_ts)-1]

    final_soc = pyo.value(model.soc_MWh[final_soc_time])
    print(f'Optimization completed. Final SoC: {final_soc} MWh')

    return df_first_day, final_soc


def battery_optimization_SRS_activation_1H(df_pred_window, previous_soc, POWER_MAX_MW=1.0, CAPACITY=3.8, ETA_CHARGE=0.9, ETA_DISCHARGE=0.9, DT_H=0.25, final_soc=0.0, beta_params=None):
    BIG_M=1e5
    df= df_pred_window.copy()

    # ----------------
    # Inicializar parámetros desde dataframe
    # ----------------
    def make_param(col):
        return {t: float(v) for t, v in zip(df['datetime'], df[col])}

    PRICE_SPOT      = make_param('SPOT')          # [€/MWh]
    PRICE_BANDA_UP  = make_param('BANDA_subir')   # [€/MW] (por QH — no multiplicar por DT_H)
    PRICE_BANDA_DN  = make_param('BANDA_bajar')   # [€/MW] (por QH)
    PRICE_ESEC_UP = make_param('SPOT')   # [€/MW] (por QH)
    PRICE_ESEC_DN = make_param('SPOT')   # [€/MW] (por QH)

    # ----------------
    # Crear modelo
    # ----------------
    model = pyo.ConcreteModel()


    time_list = df['datetime'].tolist()        
    soc_time_list = time_list + [time_list[-1] + pd.Timedelta(hours=DT_H)]

    n_periods = len(time_list)

    ### NEW SECTION → índice entero paralelo a timestamps
    T_idx = list(range(n_periods))   # 0..N-1
    soc_idx = list(range(n_periods+1))

    # Mapping índice ↔ timestamp
    idx_to_ts = {i: time_list[i] for i in T_idx}
    ts_to_idx = {ts: i for i, ts in idx_to_ts.items()}

    soc_idx_to_ts = {i: soc_time_list[i] for i in soc_idx}
    ts_to_soc_idx = {soc_time_list[i]: i for i in soc_idx}

    model.T = pyo.Set(initialize=time_list, ordered=True)          # Para potencia y precios
    model.T_soc = pyo.Set(initialize=soc_time_list, ordered=True)  # Para SOC (97 puntos)
    model.T_idx = pyo.Set(initialize=T_idx, ordered=True)
    model.T_soc_idx = pyo.Set(initialize=soc_idx, ordered=True)
    # times = list(model.T)
    # soc_times = list(model.T_soc)
    # ----------------
    # Parámetros del modelo
    # ----------------
    model.price_spot = pyo.Param(model.T, initialize=PRICE_SPOT)
    model.price_banda_up = pyo.Param(model.T, initialize=PRICE_BANDA_UP)
    model.price_banda_down = pyo.Param(model.T, initialize=PRICE_BANDA_DN)
    model.price_esec_up = pyo.Param(model.T, initialize=PRICE_ESEC_UP)
    model.price_esec_down = pyo.Param(model.T, initialize=PRICE_ESEC_DN)

    p_up, p_dn = 0.8620585457979225, 0.8420396600566572
    q_up = p_up / (p_up + p_dn)
    q_dn = 1 - q_up

    # activation_request 
    ACT_REQ_UP = {}
    ACT_REQ_DN = {}
    for t in time_list:
        if np.random.rand() < q_up:
            ACT_REQ_UP[t] = beta.rvs(beta_params['alpha_up'], beta_params['beta_up'])
            ACT_REQ_DN[t] = 0.0
        else:
            ACT_REQ_UP[t] = 0.0
            ACT_REQ_DN[t] = beta.rvs(beta_params['alpha_down'], beta_params['beta_down'])

    model.activation_request_up = pyo.Param(model.T, initialize=ACT_REQ_UP, mutable=True, within=pyo.NonNegativeReals)
    model.activation_request_down = pyo.Param(model.T, initialize=ACT_REQ_DN, mutable=True, within=pyo.NonNegativeReals)

    # ----------------
    # Variables
    # ----------------
    model.charge_base_MW = pyo.Var(model.T, bounds=(0, POWER_MAX_MW))
    model.discharge_base_MW = pyo.Var(model.T, bounds=(0, POWER_MAX_MW))
    model.uc = pyo.Var(model.T, domain=pyo.Binary)
    model.ud = pyo.Var(model.T, domain=pyo.Binary)

    model.banda_offered_up = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 2 * POWER_MAX_MW))
    model.banda_offered_down = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 2 * POWER_MAX_MW))

    model.soc_MWh = pyo.Var(model.T_soc, bounds=(0, CAPACITY))
    model.delta_soc = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.banda_up_activated = pyo.Var(model.T, bounds=(0, 2 * POWER_MAX_MW))
    model.banda_down_activated = pyo.Var(model.T, bounds=(0, 2 * POWER_MAX_MW))

    # ----------------
    # Restricciones
    # ----------------

    # Exclusividad carga/descarga (binarios)
    def exclusivity_rule(m, t):
        return m.uc[t] + m.ud[t] <= 1
    model.exclusivity = pyo.Constraint(model.T, rule=exclusivity_rule)

    # Límites base condicionados a los binarios (si uc=0 entonces carga_base debe ser 0)
    def base_charge_limit_rule(m, t):
        return m.charge_base_MW[t] <= POWER_MAX_MW * m.uc[t]
    model.base_charge_limit = pyo.Constraint(model.T, rule=base_charge_limit_rule)

    def base_discharge_limit_rule(m, t):
        return m.discharge_base_MW[t] <= POWER_MAX_MW * m.ud[t]
    model.base_discharge_limit = pyo.Constraint(model.T, rule=base_discharge_limit_rule)


    # Restricciones de bandas ofertadas
    # Caso carga (uc=1)
    def banda_up_carga_soc_rule(m, t):
        return m.banda_offered_up[t] <= m.soc_MWh[t]/DT_H + m.charge_base_MW[t] + (1 - m.uc[t]) * BIG_M
    model.banda_up_carga_soc = pyo.Constraint(model.T, rule=banda_up_carga_soc_rule)

    def banda_up_carga_max_rule(m, t):
        return m.banda_offered_up[t] <= m.charge_base_MW[t] + POWER_MAX_MW + (1 - m.uc[t]) * BIG_M
    model.banda_up_carga_max = pyo.Constraint(model.T, rule=banda_up_carga_max_rule)

    # Caso descarga (ud=1)
    def banda_up_descarga_soc_rule(m, t):
        return m.banda_offered_up[t] <= m.soc_MWh[t]/DT_H + (1 - m.ud[t]) * BIG_M
    model.banda_up_descarga_soc = pyo.Constraint(model.T, rule=banda_up_descarga_soc_rule)

    def banda_up_descarga_max_rule(m, t):
        return m.banda_offered_up[t] <= POWER_MAX_MW - m.discharge_base_MW[t] + (1 - m.ud[t]) * BIG_M
    model.banda_up_descarga_max = pyo.Constraint(model.T, rule=banda_up_descarga_max_rule)


    # Caso carga (uc=1)
    def banda_down_carga_rule(m, t):
        return m.banda_offered_down[t] <= POWER_MAX_MW - m.charge_base_MW[t] + (1 - m.uc[t]) * BIG_M
    model.banda_down_carga = pyo.Constraint(model.T, rule=banda_down_carga_rule)

    def banda_down_carga_cap_rule(m, t):
        return m.banda_offered_down[t] <= (CAPACITY - m.soc_MWh[t])/DT_H + (1 - m.uc[t]) * BIG_M
    model.banda_down_carga_cap = pyo.Constraint(model.T, rule=banda_down_carga_cap_rule)

    # Caso descarga (ud=1)
    def banda_down_descarga_rule(m, t):
        return m.banda_offered_down[t] <= m.discharge_base_MW[t] + POWER_MAX_MW + (1 - m.ud[t]) * BIG_M
    model.banda_down_descarga = pyo.Constraint(model.T, rule=banda_down_descarga_rule)

    def banda_down_descarga_cap_rule(m, t):
        return m.banda_offered_down[t] <= m.discharge_base_MW[t] + (CAPACITY - m.soc_MWh[t])/DT_H + (1 - m.ud[t]) * BIG_M
    model.banda_down_descarga_cap = pyo.Constraint(model.T, rule=banda_down_descarga_cap_rule)

    # Caso idle: ni carga ni descarga (uc=0, ud=0)
    def banda_up_idle_max_rule(m, t):
        return m.banda_offered_up[t] <= POWER_MAX_MW + BIG_M * (m.uc[t] + m.ud[t])
    model.banda_up_idle_max = pyo.Constraint(model.T, rule=banda_up_idle_max_rule)

    def banda_up_idle_soc_rule(m, t):
        return m.banda_offered_up[t] <= m.soc_MWh[t]/DT_H + BIG_M * (m.uc[t] + m.ud[t])
    model.banda_up_idle_soc = pyo.Constraint(model.T, rule=banda_up_idle_soc_rule)

    def banda_down_idle_max_rule(m, t):
        return m.banda_offered_down[t] <= POWER_MAX_MW + BIG_M*(m.uc[t] + m.ud[t])
    model.banda_down_idle_max = pyo.Constraint(model.T, rule=banda_down_idle_max_rule)

    def banda_down_idle_cap_rule(m, t):
        return m.banda_offered_down[t] <= (CAPACITY - m.soc_MWh[t])/DT_H + BIG_M*(m.uc[t] + m.ud[t])
    model.banda_down_idle_cap = pyo.Constraint(model.T, rule=banda_down_idle_cap_rule)

    # Activación aceptada por la batería <= disponibilidad ofertada
    def activation_up_rule(m, t):
        return m.banda_up_activated[t] == m.activation_request_up[t] * m.banda_offered_up[t]

    model.activation_up_con = pyo.Constraint(model.T, rule=activation_up_rule)

    def activation_down_rule(m, t):
        return m.banda_down_activated[t] == m.activation_request_down[t] * m.banda_offered_down[t]

    model.activation_down_con = pyo.Constraint(model.T, rule=activation_down_rule)
    

    # SoC inicial (MW-equivalente)
    def soc_init_rule(m):
        return m.soc_MWh[soc_time_list[0]] == previous_soc
    model.soc_init = pyo.Constraint(rule=soc_init_rule)

    # Dinámica del SoC (en MW-equivalente)
    # def soc_dynamics_rule(m, t):
    #     times = list(m.T)        # 00:00 ... 23:45
    #     soc_times = list(m.T_soc)  # 00:00 ... 23:45, 24:00

    #     idx = times.index(t)
    #     t está en posición idx en times, y en la misma posición idx en soc_times
    #     t_soc = soc_times[idx]       # mismo instante que t
    #     t_next_soc = soc_times[idx+1] 

    #     charge_net_MW = (m.charge_base_MW[t] + m.banda_down_activated[t])
    #     discharge_net_MW = (m.discharge_base_MW[t] + m.banda_up_activated[t])
    #     return m.soc_MWh[t_next_soc] == m.soc_MWh[t_soc] + (charge_net_MW * ETA_CHARGE)*DT_H - (discharge_net_MW / ETA_DISCHARGE)*DT_H
    # model.soc_dyn = pyo.Constraint(model.T, rule=soc_dynamics_rule)

    # Dinámica del SoC
    def soc_dynamics_rule(m, t):
        idx = ts_to_idx[t]
        t_soc = soc_time_list[idx]
        t_next_soc = soc_time_list[idx+1]

        charge_net_MW = (m.charge_base_MW[t] + m.banda_down_activated[t])
        discharge_net_MW = (m.discharge_base_MW[t] + m.banda_up_activated[t])

        return m.soc_MWh[t_next_soc] == m.soc_MWh[t_soc] \
               + charge_net_MW * ETA_CHARGE * DT_H \
               - discharge_net_MW / ETA_DISCHARGE * DT_H

    model.soc_dyn = pyo.Constraint(model.T, rule=soc_dynamics_rule)

    # ABS del SOC usando índices enteros
    def abs_upper(m, i):
        if i == n_periods - 1:
            return pyo.Constraint.Skip
        t = idx_to_ts[i]
        t_next = soc_time_list[i+1]
        return m.delta_soc[i] >= m.soc_MWh[t_next] - m.soc_MWh[t]

    def abs_lower(m, i):
        if i == n_periods - 1:
            return pyo.Constraint.Skip
        t = idx_to_ts[i]
        t_next = soc_time_list[i+1]
        return m.delta_soc[i] >= -(m.soc_MWh[t_next] - m.soc_MWh[t])

    model.abs_upper = pyo.Constraint(model.T_idx, rule=abs_upper)
    model.abs_lower = pyo.Constraint(model.T_idx, rule=abs_lower)

    timesteps = 96
    # ≈≈≈ RESTRICCIÓN CICLOS ≈≈≈  
    n_days = n_periods // timesteps
    day_slices = [list(range(d*timesteps, (d+1)*timesteps)) for d in range(n_days)]

    def limit_cycles(m, d):
        return sum(m.delta_soc[i] for i in day_slices[d]) <= 4 * CAPACITY

    model.daily_cycle_limit = pyo.Constraint(range(n_days), rule=limit_cycles)

    # ----------------
    # Objetivo: valor esperado (€/día)
    # ----------------
    def obj_rule(m):
        total = sum(
            m.price_spot[t] * (m.discharge_base_MW[t] - m.charge_base_MW[t]) * DT_H
            + m.price_banda_up[t] * m.banda_offered_up[t]
            + m.price_banda_down[t] * m.banda_offered_down[t]
            + m.price_esec_up[t] * m.banda_up_activated[t]*DT_H
            - m.price_esec_down[t] * m.banda_down_activated[t]*DT_H
            for t in m.T
        )
        return total
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # ----------------
    # Resolver
    # ----------------
    solver = SolverFactory('gurobi')
    res = solver.solve(model, tee=True)
    print("Solver termination:", res.solver.termination_condition)

    # ----------------
    # Resultados y CSV final (incluye activation_request y SoC inicio/fin)
    # ----------------
    rows = []
    for t in model.T:
        charge_base = pyo.value(model.charge_base_MW[t])
        discharge_base = pyo.value(model.discharge_base_MW[t])
        uc = pyo.value(model.uc[t])
        ud = pyo.value(model.ud[t])
        banda_up = pyo.value(model.banda_offered_up[t])
        banda_down = pyo.value(model.banda_offered_down[t])
        act_req_up = pyo.value(model.activation_request_up[t])
        act_req_down = pyo.value(model.activation_request_down[t])
        banda_up_activated = pyo.value(model.banda_up_activated[t])
        banda_down_activated = pyo.value(model.banda_down_activated[t])

        # SoC inicio (MWh)
        soc_start_MWh = pyo.value(model.soc_MWh[t])

        # Valores económicos por fila (€/QH)
        spot_term = model.price_spot[t] * (discharge_base - charge_base)*DT_H
        banda_term = model.price_banda_up[t] * banda_up + model.price_banda_down[t] * banda_down
        #esec_cost = (model.price_esec_up[t] * banda_up_activated - model.price_esec_down[t] * banda_down_activated) * DT_H
        net = spot_term + banda_term

        rows.append({
            'datetime': t,
            'uc': abs(uc),
            'ud': abs(ud),
            'charge_base_MW': charge_base,
            'discharge_base_MW': discharge_base,
            'banda_up_offered_MW': banda_up,
            'banda_down_offered_MW': banda_down,
            'activation_request_up_MW': act_req_up,
            'activation_request_down_MW': act_req_down,
            'banda_up_activated_MW': banda_up_activated,
            'banda_down_activated_MW': banda_down_activated,
            'soc_start_MWh': soc_start_MWh,
            'spot_revenue_EUR': spot_term,
            'banda_revenue_EUR': banda_term,
            'net_profit_EUR': net
        })

    out = pd.DataFrame(rows).sort_values('datetime').reset_index(drop=True)
    df_first_day = out.iloc[:96]
    
    if timesteps < len(soc_idx_to_ts):
        final_soc_time = soc_idx_to_ts[timesteps]   # timestamp del SOC en t=192        
    else:
        # horizonte más corto (muy raro)
        final_soc_time = soc_idx_to_ts[len(soc_idx_to_ts)-1]

    final_soc = pyo.value(model.soc_MWh[final_soc_time])
    print(f'Optimization completed. Final SoC: {final_soc} MWh')

    return df_first_day, final_soc

def battery_optimization_baseline(df_pred_window, previous_soc, POWER_MAX_MW=1.0, CAPACITY=3.8, ETA_CHARGE=1.0, ETA_DISCHARGE=1.0, DT_H=0.25, final_soc=0.0):
    BIG_M = 1e5
    df = df_pred_window.copy()

    # ----------------
    # Inicializar parámetros desde dataframe
    # ----------------
    def make_param(col):
        return {t: float(v) for t, v in zip(df['datetime'], df[col])}

    PRICE_SPOT = make_param('SPOT')          # [€/MWh]

    # ----------------
    # Crear modelo
    # ----------------
    model = pyo.ConcreteModel()

    # Set de tiempos (por fechas)
    model.T = pyo.Set(initialize=df['datetime'].tolist(), ordered=True)
    times = list(model.T)

    # ----------------
    # Parámetros del modelo
    # ----------------
    model.price_spot = pyo.Param(model.T, initialize=PRICE_SPOT)

    # ----------------
    # Variables
    # ----------------
    model.charge_base_MW = pyo.Var(model.T, bounds=(0, POWER_MAX_MW))
    model.discharge_base_MW = pyo.Var(model.T, bounds=(0, POWER_MAX_MW))
    model.uc = pyo.Var(model.T, domain=pyo.Binary)
    model.ud = pyo.Var(model.T, domain=pyo.Binary)

    model.soc_MWh = pyo.Var(model.T, bounds=(0, CAPACITY))

    # ----------------
    # Restricciones
    # ----------------

    # Exclusividad carga/descarga (binarios)
    def exclusivity_rule(m, t):
        return m.uc[t] + m.ud[t] <= 1
    model.exclusivity = pyo.Constraint(model.T, rule=exclusivity_rule)

    # Límites base condicionados a los binarios (si uc=0 entonces carga_base debe ser 0)
    def base_charge_limit_rule(m, t):
        return m.charge_base_MW[t] <= POWER_MAX_MW * m.uc[t]
    model.base_charge_limit = pyo.Constraint(model.T, rule=base_charge_limit_rule)

    def base_discharge_limit_rule(m, t):
        return m.discharge_base_MW[t] <= POWER_MAX_MW * m.ud[t]
    model.base_discharge_limit = pyo.Constraint(model.T, rule=base_discharge_limit_rule)


    # SoC inicial y final (MW-equivalente)
    def soc_init_rule(m):
        return m.soc_MWh[times[0]] == previous_soc
    model.soc_init = pyo.Constraint(rule=soc_init_rule)

    # Dinámica del SoC (en MW-equivalente)
    def soc_dynamics_rule(m, t):
        t_list = list(m.T)
        idx = t_list.index(t)
        if idx == len(t_list) - 1:
            return pyo.Constraint.Skip
        t_next = t_list[idx + 1]
        charge_net_MW = (m.charge_base_MW[t])
        discharge_net_MW = (m.discharge_base_MW[t])
        return m.soc_MWh[t_next] == m.soc_MWh[t] + (charge_net_MW * ETA_CHARGE)*DT_H - (discharge_net_MW / ETA_DISCHARGE)*DT_H
    model.soc_dyn = pyo.Constraint(model.T, rule=soc_dynamics_rule)

    # ABS del SOC usando índices enteros
    def abs_upper(m, i):
        if i == n_periods - 1:
            return pyo.Constraint.Skip
        t = idx_to_ts[i]
        t_next = soc_time_list[i+1]
        return m.delta_soc[i] >= m.soc_MWh[t_next] - m.soc_MWh[t]

    def abs_lower(m, i):
        if i == n_periods - 1:
            return pyo.Constraint.Skip
        t = idx_to_ts[i]
        t_next = soc_time_list[i+1]
        return m.delta_soc[i] >= -(m.soc_MWh[t_next] - m.soc_MWh[t])

    model.abs_upper = pyo.Constraint(model.T_idx, rule=abs_upper)
    model.abs_lower = pyo.Constraint(model.T_idx, rule=abs_lower)

    # ≈≈≈ RESTRICCIÓN CICLOS ≈≈≈  
    n_days = n_periods // timesteps
    day_slices = [list(range(d*timesteps, (d+1)*timesteps)) for d in range(n_days)]

    def limit_cycles(m, d):
        return sum(m.delta_soc[i] for i in day_slices[d]) <= 4 * CAPACITY

    model.daily_cycle_limit = pyo.Constraint(range(n_days), rule=limit_cycles)
    
    # ----------------
    # Objetivo: valor esperado (€/día)
    # ----------------
    def obj_rule(m):
        total = sum(
            m.price_spot[t] * (m.discharge_base_MW[t] - m.charge_base_MW[t]) * DT_H
            for t in m.T
        )
        return total
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # ----------------
    # Resolver
    # ----------------
    solver = SolverFactory('gurobi')
    res = solver.solve(model, tee=False)
    print("Solver termination:", res.solver.termination_condition)

    # ----------------
    # Resultados y CSV final (incluye activation_request y SoC inicio/fin)
    # ----------------
    rows = []
    for t in model.T:
        charge_base = pyo.value(model.charge_base_MW[t])
        discharge_base = pyo.value(model.discharge_base_MW[t])
        uc = pyo.value(model.uc[t])
        ud = pyo.value(model.ud[t])

        # SoC inicio (MWh)
        soc_start_MWh = pyo.value(model.soc_MWh[t])

        # Valores económicos por fila (€/QH)
        spot_term = model.price_spot[t] * (discharge_base - charge_base)*DT_H

        rows.append({
            'datetime': t,
            'uc': abs(uc),
            'ud': abs(ud),
            'charge_base_MW': charge_base,
            'discharge_base_MW': discharge_base,
            'soc_start_MWh': soc_start_MWh,
            'spot_revenue_EUR': spot_term,
            'net_profit_EUR': spot_term
        })

    out = pd.DataFrame(rows).sort_values('datetime').reset_index(drop=True)
    df_first_day = out.iloc[:96]

    if len(out) > 96:
        final_soc = out['soc_start_MWh'].iloc[96]
    else:
        # final_soc = out['soc_start_MWh'].iloc[95]
        # - (pyo.value(model.discharge_base_MW[times[95]]) / ETA_DISCHARGE)*DT_H   
        final_soc = out['soc_start_MWh'].iloc[95]

    print(f'Optimization completed. Final SoC: {final_soc} MWh')

    return df_first_day, final_soc

def battery_optimization_baseline_1H(
    df_pred_window,
    previous_soc,
    POWER_MAX_MW=1.0,
    CAPACITY=3.8,
    ETA_CHARGE=0.9,
    ETA_DISCHARGE=0.9,
    DT_H=0.25,
    final_soc=0.0,
    timesteps = 96, 
    beta_params=None
):
    BIG_M = 1e5
    df = df_pred_window.copy()

    # ----------------
    # Inicializar parámetros desde dataframe
    # ----------------
    def make_param(col):
        return {t: float(v) for t, v in zip(df['datetime'], df[col])}

    PRICE_SPOT = make_param('SPOT')          # [€/MWh]

    # ======================================================
    # CONJUNTOS DE TIEMPO
    # ======================================================

    time_list = df['datetime'].tolist()        
    soc_time_list = time_list + [time_list[-1] + pd.Timedelta(hours=DT_H)]

    n_periods = len(time_list)

    ### NEW SECTION → índice entero paralelo a timestamps
    T_idx = list(range(n_periods))   # 0..N-1
    soc_idx = list(range(n_periods+1))

    # Mapping índice ↔ timestamp
    idx_to_ts = {i: time_list[i] for i in T_idx}
    ts_to_idx = {ts: i for i, ts in idx_to_ts.items()}

    soc_idx_to_ts = {i: soc_time_list[i] for i in soc_idx}
    ts_to_soc_idx = {soc_time_list[i]: i for i in soc_idx}

    # ----------------
    # Crear modelo
    # ----------------
    model = pyo.ConcreteModel()

    # Sets originales (timestamps)
    model.T = pyo.Set(initialize=time_list, ordered=True)
    model.T_soc = pyo.Set(initialize=soc_time_list, ordered=True)

    # Sets nuevos (índices numéricos)
    model.T_idx = pyo.Set(initialize=T_idx, ordered=True)
    model.T_soc_idx = pyo.Set(initialize=soc_idx, ordered=True)

    # ----------------
    # Parámetros del modelo
    # ----------------
    model.price_spot = pyo.Param(model.T, initialize=PRICE_SPOT)

    # ----------------
    # Variables indexadas por timestamps
    # ----------------
    model.charge_base_MW = pyo.Var(model.T, bounds=(0, POWER_MAX_MW))
    model.discharge_base_MW = pyo.Var(model.T, bounds=(0, POWER_MAX_MW))
    model.uc = pyo.Var(model.T, domain=pyo.Binary)
    model.ud = pyo.Var(model.T, domain=pyo.Binary)

    model.soc_MWh = pyo.Var(model.T_soc, bounds=(0, CAPACITY))

    # ----------------
    # NUEVA delta_soc indexada por ENTEROS
    # ----------------
    model.delta_soc = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)

    # ----------------
    # Restricciones
    # ----------------

    # Exclusividad carga/descarga
    def exclusivity_rule(m, t):
        return m.uc[t] + m.ud[t] <= 1
    model.exclusivity = pyo.Constraint(model.T, rule=exclusivity_rule)

    def base_charge_limit_rule(m, t):
        return m.charge_base_MW[t] <= POWER_MAX_MW * m.uc[t]
    model.base_charge_limit = pyo.Constraint(model.T, rule=base_charge_limit_rule)

    def base_discharge_limit_rule(m, t):
        return m.discharge_base_MW[t] <= POWER_MAX_MW * m.ud[t]
    model.base_discharge_limit = pyo.Constraint(model.T, rule=base_discharge_limit_rule)

    # SoC inicial
    def soc_init_rule(m):
        return m.soc_MWh[soc_time_list[0]] == previous_soc
    model.soc_init = pyo.Constraint(rule=soc_init_rule)

    # Dinámica del SoC
    def soc_dynamics_rule(m, t):
        idx = ts_to_idx[t]
        t_soc = soc_time_list[idx]
        t_next_soc = soc_time_list[idx+1]

        charge = m.charge_base_MW[t]
        discharge = m.discharge_base_MW[t]

        return m.soc_MWh[t_next_soc] == m.soc_MWh[t_soc] \
               + charge * ETA_CHARGE * DT_H \
               - discharge / ETA_DISCHARGE * DT_H

    model.soc_dyn = pyo.Constraint(model.T, rule=soc_dynamics_rule)

    # ABS del SOC usando índices enteros
    def abs_upper(m, i):
        if i == n_periods - 1:
            return pyo.Constraint.Skip
        t = idx_to_ts[i]
        t_next = soc_time_list[i+1]
        return m.delta_soc[i] >= m.soc_MWh[t_next] - m.soc_MWh[t]

    def abs_lower(m, i):
        if i == n_periods - 1:
            return pyo.Constraint.Skip
        t = idx_to_ts[i]
        t_next = soc_time_list[i+1]
        return m.delta_soc[i] >= -(m.soc_MWh[t_next] - m.soc_MWh[t])

    model.abs_upper = pyo.Constraint(model.T_idx, rule=abs_upper)
    model.abs_lower = pyo.Constraint(model.T_idx, rule=abs_lower)

    # ≈≈≈ RESTRICCIÓN CICLOS ≈≈≈  
    n_days = n_periods // timesteps
    day_slices = [list(range(d*timesteps, (d+1)*timesteps)) for d in range(n_days)]

    def limit_cycles(m, d):
        return sum(m.delta_soc[i] for i in day_slices[d]) <= 4 * CAPACITY

    model.daily_cycle_limit = pyo.Constraint(range(n_days), rule=limit_cycles)

    # ----------------
    # Objetivo: valor esperado (€/día)
    # ----------------
    def obj_rule(m):
        total = sum(
            m.price_spot[t] * (m.discharge_base_MW[t] - m.charge_base_MW[t]) * DT_H
            for t in m.T
        )
        return total
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # ----------------
    # Resolver
    # ----------------
    solver = SolverFactory('gurobi')
    res = solver.solve(model, tee=False)
    print("Solver termination:", res.solver.termination_condition)

    # ----------------
    # Resultados y CSV final
    # ----------------
    rows = []
    for t in model.T:
        charge_base = pyo.value(model.charge_base_MW[t])
        discharge_base = pyo.value(model.discharge_base_MW[t])
        uc = pyo.value(model.uc[t])
        ud = pyo.value(model.ud[t])

        # SoC al inicio del intervalo (MWh)
        soc_start_MWh = pyo.value(model.soc_MWh[t])

        # Valores económicos por fila (€/QH)
        spot_term = model.price_spot[t] * (discharge_base - charge_base) * DT_H

        rows.append({
            'datetime': t,
            'uc': abs(uc),
            'ud': abs(ud),
            'charge_base_MW': charge_base,
            'discharge_base_MW': discharge_base,
            'soc_start_MWh': soc_start_MWh,
            'spot_revenue_EUR': spot_term,
            'net_profit_EUR': spot_term
        })

    out = pd.DataFrame(rows).sort_values('datetime').reset_index(drop=True)

    df_first_day = out.iloc[:96]

    if timesteps < len(soc_idx_to_ts):
        final_soc_time = soc_idx_to_ts[timesteps]         
    else:
        final_soc_time = soc_idx_to_ts[len(soc_idx_to_ts)-1]

    final_soc = pyo.value(model.soc_MWh[final_soc_time])

    print(f"Optimization completed. Final SoC: {final_soc} MWh")

    return df_first_day, final_soc


def update_revenue_from_predicted_decisions(df_decision, df_real_prices, DT_H=0.25):
    df = df_decision.copy()
    real = df_real_prices.reset_index(drop=True)
    df = df.reset_index(drop=True)

    # Spot revenue
    df["spot_revenue_EUR"] = real["SPOT"] * \
        (df["discharge_base_MW"] - df["charge_base_MW"]) * DT_H

    # Si hay columnas de banda, añadir revenue de banda

    if {"esec_cost_EUR"}.issubset(df.columns):
        df["banda_revenue_EUR"] = (
            real["BANDA_subir"] * df.get("banda_up_offered_MW", 0)
            +
            real["BANDA_bajar"] * df.get("banda_down_offered_MW", 0)
        )

        df["esec_cost_EUR"] = real["ESEC_subir"] * df.get("banda_up_activated_MW", 0) * DT_H - \
            real["ESEC_bajar"] * df.get("banda_down_activated_MW", 0) * DT_H
        
        df["net_profit_EUR"] = df["spot_revenue_EUR"] + df["esec_cost_EUR"] + df["banda_revenue_EUR"]
    elif {"banda_up_offered_MW", "banda_down_offered_MW"}.issubset(df.columns):
        df["banda_revenue_EUR"] = (
            real["BANDA_subir"] * df.get("banda_up_offered_MW", 0)
            +
            real["BANDA_bajar"] * df.get("banda_down_offered_MW", 0)
        )

        df["net_profit_EUR"] = df["spot_revenue_EUR"] + df["banda_revenue_EUR"]
    else:
        df["net_profit_EUR"] = df["spot_revenue_EUR"]

    return df


import pandas as pd

def compare_optimizations(df_expost, df_exante, df_perfect_foresight, horizon, model_type):

    def prep(df, strategy):
        df = df.copy()
        df['day'] = df['datetime'].dt.date

        # Beneficio diario
        df = df.groupby('day', as_index=False)['net_profit_EUR'].sum()

        # Añadimos info
        df['Strategy'] = strategy
        df['Horizon'] = horizon
        df['Model'] = model_type

        # Nos quedamos solo con columnas necesarias
        df = df[['day', 'net_profit_EUR', 'Strategy', 'Horizon', 'Model']]

        return df

    df_expost_revenue = prep(df_expost, 'Ex-post')
    df_exante_revenue = prep(df_exante, 'Ex-ante')
    df_pf_revenue = prep(df_perfect_foresight, 'Perfect Foresight')

    df_net_global_profit = pd.concat(
        [df_exante_revenue, df_expost_revenue, df_pf_revenue],
        ignore_index=True
    )
    
    return df_net_global_profit

