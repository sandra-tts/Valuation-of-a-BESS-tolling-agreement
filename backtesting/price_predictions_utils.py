import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

def fill_future_lags(df_full, df_day, col, activation, timesteps, max_lag_days=7):
    max_lag_days = 2 if activation else max_lag_days

    for day_lag in range(1, max_lag_days + 1):
        # Índices del df_full que quieres usar:
        start = -(timesteps * day_lag)
        end = None if day_lag == 1 else -(timesteps * (day_lag - 1))

        # Extraer los valores correspondientes a ese lag
        lag_values = df_full[col].iloc[start:end].values

        # Asegurar que la longitud coincide (por seguridad)
        if len(lag_values) != len(df_day):
            raise ValueError(f"Error: expected {len(df_day)} values, got {len(lag_values)}")

        # Asignar columna lag al df_day
        df_day[f"lag_day_{day_lag}_hour"] = lag_values

    return df_day

def add_cyclical_time_features(df, timestamp_col='datetime'):
    df = df.copy()

    df['hour'] = df[timestamp_col].dt.hour
    df['minute'] = df[timestamp_col].dt.minute
    df['time_decimal'] = df['hour'] + df['minute'] / 60.0
    df['day_of_week'] = df[timestamp_col].dt.dayofweek

    # Daily cycle (24h)
    radians_day = 2 * np.pi * df['time_decimal'] / 24
    df['tod_sin'] = np.sin(radians_day)
    df['tod_cos'] = np.cos(radians_day)

    # Weekly cycle (7 days)
    radians_week = 2 * np.pi * df['day_of_week'] / 7
    df['dow_sin'] = np.sin(radians_week)
    df['dow_cos'] = np.cos(radians_week)
    
    return df

def generate_future_day_rows(current_day, activation=False, timestep="15min"):
    start_dt = pd.Timestamp(current_day)
    end_dt = start_dt + pd.Timedelta(days=1)

# generar timestamps
    future_times = pd.date_range(start=start_dt, end=end_dt, freq=timestep)
    df_future = pd.DataFrame({"datetime": future_times})
    df_future = add_cyclical_time_features(df_future)
    df_future['is_weekend'] = df_future['day_of_week'].isin([5,6]).astype(int)
    df_future['is_holiday'] = df_future['datetime'].dt.date.isin(
        pd.to_datetime(['2024-12-25', '2025-01-01', '2025-04-18', '2025-05-01', '2025-08-15']).date
    ).astype(int)
    if activation:
        df_future['is_ramp_hour'] = df_future['hour'].isin([8,9,10,19,20,21]).astype(int)
    else: 
        df_future['is_peak_hour'] = df_future['hour'].isin([8,9,18,19]).astype(int)

    return df_future

import pandas as pd

def convert_hourly_to_qh(df, cutoff="2025-10-01"):
    cutoff_date = pd.to_datetime(cutoff).date()

    # Separar
    df_before = df[df["datetime"].dt.date < cutoff_date].copy()
    df_after  = df[df["datetime"].dt.date >= cutoff_date].copy()

    if not df_before.empty:
        df_before = df_before.set_index("datetime")

        # Última fecha antes del corte
        last_ts = df_before.index.max()

        # Queremos completar hasta las 23:45 del mismo día
        end_of_day = last_ts.normalize() + pd.Timedelta(hours=23, minutes=45)

        # Si falta completar el día, añadimos un dummy
        if last_ts < end_of_day:
            df_before.loc[end_of_day] = df_before.iloc[-1]

        # Ahora sí resample completo
        df_before = df_before.resample("15min").ffill().reset_index()

    # Unir y ordenar
    df_final = pd.concat([df_before, df_after], ignore_index=True)
    df_final = df_final.sort_values("datetime").reset_index(drop=True)

    return df_final
