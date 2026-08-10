import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from price_predictions_utils import convert_hourly_to_qh, generate_future_day_rows, fill_future_lags, add_cyclical_time_features
import xgboost as xgb
import os
from esios import ESIOSClient

def generate_model_parameters(start_date = '2024-11-20', split_date = '2025-09-01', end_date ='2025-11-04'):
# ponemos como end_date un día más del que realmente hacemos predicción
    token = os.getenv("ESIOS_TOKEN")
    if token is None:
        raise ValueError("ESIOS_TOKEN no está definido en las variables de entorno")

    client = ESIOSClient(token)
    endpoint = client.endpoint(name='indicators')

    results = {} 

    # -------------------------------------------------------
    # PROCESAR INDICADORES BANDA DE MERCADO

    indicators = [2130, 634]  # IDs ESIOS
    nombres_indicadores = ['BANDA_subir', 'BANDA_bajar']

    for idx, indicator_id in enumerate(indicators):
        print("\n" + "="*70)
        print(f"Procesando indicador: {nombres_indicadores[idx]} ({indicator_id})")
        print("="*70)

        df_price = endpoint.select(id=indicator_id).historical(start=start_date, end=end_date)
        df_price = pd.DataFrame(df_price).reset_index()
        df_price['datetime'] = pd.to_datetime(df_price['datetime'])

        df_price = df_price.drop(columns=['geo_name', 'geo_id'], errors='ignore')
        # Eliminar cambio de hora
        clock_change_winter = pd.to_datetime([
            "2025-10-26 02:00:00+02:00",
            "2025-10-26 02:15:00+02:00",
            "2025-10-26 02:30:00+02:00",
            "2025-10-26 02:45:00+02:00"
        ])

        df_price = df_price[~df_price['datetime'].isin(clock_change_winter)]
        change_dates = ["2025-03-30"]
        df_price = df_price[~df_price["datetime"].dt.date.astype(str).isin(change_dates)]
        df = df_price.copy()
        df = add_cyclical_time_features(df)
        
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        df['is_holiday'] = df['datetime'].dt.date.isin(
            pd.to_datetime(['2024-12-25', '2025-01-01', '2025-04-18', '2025-05-01', '2025-08-15']).date
        ).astype(int)
        df['is_peak_hour'] = df['hour'].isin([8,9,18,19]).astype(int)

        # Frecuencia de datos: 96 puntos/día (15 min)
        for day_lag in range(1,8):
            df[f'lag_day_{day_lag}_hour'] = df[str(indicator_id)].shift(96 * day_lag)
        df = df.drop(columns=['hour', 'minute', 'time_decimal', 'day_of_week'])
        df = df.dropna().reset_index(drop=True)

        train_df = df[df['datetime'] <= split_date][:-1]
        test_df = df[(df['datetime'] >= (pd.to_datetime(split_date)).strftime("%Y-%m-%d")) & (df['datetime'] < end_date)]

        print(f"Training: {train_df['datetime'].min()} → {train_df['datetime'].max()}")
        print(f"Test: {test_df['datetime'].min()} → {test_df['datetime'].max()}")
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        X_train = train_df.drop(columns=['datetime', str(indicator_id)])
        y_train = train_df[str(indicator_id)]

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # -------------------------------------------------------
        # ENTRENAR MODELO FINAL (puedes usar aquí tus mejores params del GridSearch)
        # -------------------------------------------------------
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8
        )
        model.fit(X_train_scaled, y_train)

        results[indicator_id] = {
        "train_df": train_df,
        "test_df": test_df,
        "df": df,
        "scaler": scaler,
        "model": model
        }
    # -------------------------------------------------------
    # PROCESAR INDICADOR SPOT   

    indicator_id = 600
    spot_col = str(indicator_id)
    nombre_indicador = "SPOT"

    print("\n" + "="*70)
    print(f"Procesando indicador: {nombre_indicador} ({indicator_id})")
    print("="*70)

    # -------------------------------------------------------------
    # 1. DESCARGA Y PREPARACIÓN DE DATOS
    # -------------------------------------------------------------
    df_price_spot = endpoint.select(id=indicator_id).historical(start=start_date, end=end_date)
    df_price_spot = pd.DataFrame(df_price_spot).reset_index()
    df_price_spot["datetime"] = pd.to_datetime(df_price_spot["datetime"])

    # Filtrar geografía
    df_price_spot = df_price_spot[df_price_spot["geo_id"] == 3]
    df_price_spot = df_price_spot.drop(columns=["geo_name", "geo_id"], errors="ignore")
    # Eliminar cambio horario
    df_price_spot = df_price_spot[~df_price_spot['datetime'].isin(clock_change_winter)]
    change_dates = ["2025-03-30"]
    df_price_spot = df_price_spot[
        ~df_price_spot["datetime"].dt.date.astype(str).isin(change_dates)
    ]

    # -------------------------------------------------------------
    # 2. CREACIÓN DE FEATURES
    # -------------------------------------------------------------
    df_spot = df_price_spot.copy()
    df_spot = add_cyclical_time_features(df_spot)

    df_spot["is_weekend"] = (df_spot["day_of_week"].isin([5, 6])).astype(int)

    df_spot["is_holiday"] = df_spot["datetime"].dt.date.isin(
        pd.to_datetime([
            "2024-12-25", "2025-01-01","2025-04-18",
            "2025-05-01","2025-08-15"
        ]).date
    ).astype(int)

    df_spot["is_peak_hour"] = df_spot["hour"].isin([8,9,10,19,20,21]).astype(int)
    df_spot = convert_hourly_to_qh(df_spot)
    
    # LAGS (estos solo sirven en el train inicial)
    for day_lag in range(1,8):
        df_spot[f'lag_day_{day_lag}_hour'] = df_spot['600'].shift(96 * day_lag)

    df_spot = df_spot.drop(columns=['hour', 'minute', 'time_decimal', 'day_of_week'])
    df_spot = df_spot.dropna().reset_index(drop=True)
    

    # -------------------------------------------------------------
    # 3. TRAIN / TEST SPLIT
    # -------------------------------------------------------------
    train_df_spot = df_spot[df_spot['datetime'] <= split_date][:-1]
    test_df_spot  = df_spot[(df_spot['datetime'] >= (pd.to_datetime(split_date)).strftime("%Y-%m-%d")) & (df_spot['datetime'] < end_date)]

    print(f"Train size: {len(train_df_spot)}, Test size: {len(test_df_spot)}")

    # -------------------------------------------------------------
    # 4. ESCALADO Y ENTRENAMIENTO MODELO
    # -------------------------------------------------------------
    X_train = train_df_spot.drop(columns=["datetime", spot_col])
    y_train = train_df_spot[spot_col]

    scaler_spot = MinMaxScaler()
    X_train_scaled = scaler_spot.fit_transform(X_train)

    model_spot = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model_spot.fit(X_train_scaled, y_train)

    results[indicator_id] = {
        "train_df": train_df_spot,
        "test_df": test_df_spot,
        "df": df_spot,
        "scaler": scaler_spot,
        "model": model_spot
    }

    indicators_esec = [682, 683]  # IDs ESIOS
    nombres_indicadores_esec = ['ESEC_subir', 'ESEC_bajar']

    for idx, indicator_id in enumerate(indicators_esec):
        print("\n" + "="*70)
        print(f"Procesando indicador: {nombres_indicadores[idx]} ({indicator_id})")
        print("="*70)

        df_price = endpoint.select(id=indicator_id).historical(start=start_date, end=end_date)
        df_price = pd.DataFrame(df_price).reset_index()
        df_price['datetime'] = pd.to_datetime(df_price['datetime'])

        df_price = df_price.drop(columns=['geo_name', 'geo_id'], errors='ignore')
        # Eliminar cambio de hora
        clock_change_winter = pd.to_datetime([
            "2025-10-26 02:00:00+02:00",
            "2025-10-26 02:15:00+02:00",
            "2025-10-26 02:30:00+02:00",
            "2025-10-26 02:45:00+02:00"
        ])

        df_price = df_price[~df_price['datetime'].isin(clock_change_winter)]
        change_dates = ["2025-03-30"]
        df_price = df_price[~df_price["datetime"].dt.date.astype(str).isin(change_dates)]
        df = df_price.copy()
        df = add_cyclical_time_features(df)
        
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        df['is_holiday'] = df['datetime'].dt.date.isin(
            pd.to_datetime(['2024-12-25', '2025-01-01', '2025-04-18', '2025-05-01', '2025-08-15']).date
        ).astype(int)
        df['is_ramp_hour'] = df['hour'].isin([5,6,7,8,18,19,20]).astype(int)

        # Frecuencia de datos: 96 puntos/día (15 min)
        for day_lag in range(1,3):
            df[f'lag_day_{day_lag}_hour'] = df[str(indicator_id)].shift(96 * day_lag)
        df = df.drop(columns=['hour', 'minute', 'time_decimal', 'day_of_week'])
        df = df.dropna().reset_index(drop=True)

        train_df = df[df['datetime'] <= split_date][:-1]
        test_df = df[(df['datetime'] >= (pd.to_datetime(split_date)).strftime("%Y-%m-%d")) & (df['datetime'] < end_date)]

        print(f"Training: {train_df['datetime'].min()} → {train_df['datetime'].max()}")
        print(f"Test: {test_df['datetime'].min()} → {test_df['datetime'].max()}")
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        X_train = train_df.drop(columns=['datetime', str(indicator_id)])
        y_train = train_df[str(indicator_id)]

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # -------------------------------------------------------
        # ENTRENAR MODELO FINAL (puedes usar aquí tus mejores params del GridSearch)
        # -------------------------------------------------------
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8
        )
        model.fit(X_train_scaled, y_train)

        results[indicator_id] = {
        "train_df": train_df,
        "test_df": test_df,
        "df": df,
        "scaler": scaler,
        "model": model
        }

    return results

def generate_price_predictions(model_type, horizon, current_day, model_params, dict_test_vs_pred, start_date = '2024-11-20', split_date = '2025-08-31'):
    
    if model_type == 'baseline':
        model_params = {k: v for k, v in model_params.items() if k not in [2130, 634]}

    dict_price_pred_real = {}
    dict_optimization_input = {}

    indicator_config = {
    600:  {"nombre": "SPOT",          "timesteps": 96},
    2130: {"nombre": "BANDA_subir",   "timesteps": 96},
    634:  {"nombre": "BANDA_bajar",   "timesteps": 96},
    682:  {"nombre": "ESEC_subir",    "timesteps": 96},
    683:  {"nombre": "ESEC_bajar",    "timesteps": 96},
    }
    
    print(f'Generating predictions for day {(current_day + pd.Timedelta(days=1)).date()} with horizon {horizon} days for model {model_type}.')


    for indicator_id in model_params.keys():
        activation = True if indicator_id == 682 or indicator_id == 683 else False
        col = str(indicator_id)
        df_prices = pd.DataFrame() if pd.to_datetime(split_date) == current_day else dict_test_vs_pred[col]
        dict_test_vs_pred[col] = pd.DataFrame() if pd.to_datetime(split_date) == current_day else dict_test_vs_pred[col]

        nombre_indicador = indicator_config[indicator_id]["nombre"]
        timesteps = indicator_config[indicator_id]["timesteps"]

        model = model_params[indicator_id]["model"]
        scaler  = model_params[indicator_id]["scaler"]
        train_df = model_params[indicator_id]["train_df"]
        test_df  = model_params[indicator_id]["test_df"]
        df       = model_params[indicator_id]["df"]
        # -------------------------------------------------------------
        # 5. ROLLING HORIZON AUTORREGRESIVO — DÍA A DÍA
        # -------------------------------------------------------------

        # Este dataframe irá creciendo con predicciones
        df_full = train_df.copy()
        df_full = pd.concat([df_full, df_prices], ignore_index=True)
        test_days = sorted(test_df["datetime"].dt.date.unique())
        
        if (pd.to_datetime(current_day) + pd.Timedelta(days=horizon)).date() > test_days[-1]:
            horizon = (test_days[-1] - current_day.date()).days

        target_days = [(current_day + pd.Timedelta(days=i)).date()
               for i in range(1, horizon + 1)]

        predictions_for_opt = []
        real_for_opt = []

        for idx_day, day in enumerate(target_days, start=1):
            df_day = generate_future_day_rows(day, activation)
            if day == date(2025, 10, 26): # cambio de hora
                df_day["datetime"] = pd.to_datetime(df_day["datetime"]).dt.tz_localize(
                    "Europe/Madrid",
                    ambiguous="NaT",
                    nonexistent="shift_forward"
                )

                new_times = [
                    pd.Timestamp("2025-10-26 02:00:00").tz_localize("Europe/Madrid", ambiguous=False),
                    pd.Timestamp("2025-10-26 02:15:00").tz_localize("Europe/Madrid", ambiguous=False),
                    pd.Timestamp("2025-10-26 02:30:00").tz_localize("Europe/Madrid", ambiguous=False),
                    pd.Timestamp("2025-10-26 02:45:00").tz_localize("Europe/Madrid", ambiguous=False),
                ]

                nat_idx = df_day[df_day["datetime"].isna()].index

                df_day.loc[nat_idx, "datetime"] = new_times

            else: 
                df_day["datetime"] = (pd.to_datetime(df_day["datetime"]).dt.tz_localize("Europe/Madrid"))
            df_day = df_day[:timesteps].copy()
            df_day = fill_future_lags(df_full, df_day, col, activation, timesteps = timesteps, max_lag_days = 7)
            df_day = df_day.drop(columns=['hour', 'minute', 'time_decimal', 'day_of_week'])

            if target_days[0] == day:
                df_day2 = df_day.copy()
                df_merged = (
                    df_day2
                    .merge(
                        test_df.loc[test_df["datetime"].dt.date == day, ["datetime", col]],
                        on="datetime",
                        how="left"
                    )
                )
                if df_merged[col].isna().any():
                    df_merged[col] = df_merged[col].ffill()

                df_day2[col] = df_merged[col].values
               
                dict_test_vs_pred[col] = pd.concat(
                    [dict_test_vs_pred[col], df_day2],
                    ignore_index=True
                )  # actualizamos el df_prices para el siguiente current_day
            
            df_full = pd.concat([df_full, df_day], ignore_index=True)

            X_day = df_full[df_full["datetime"].dt.date == day]
            X_day_model = X_day.drop(columns=["datetime", col])

            X_scaled = scaler.transform(X_day_model)
            y_pred = model.predict(X_scaled)
            df_full.loc[X_day.index, col] = y_pred
            
            df_day_result = pd.DataFrame({
                "datetime": X_day["datetime"],
                "pred": y_pred,
                "real": df_merged[col].values
            })
            
            df_day_result["percentage_error"] = np.where(
                df_day_result["real"] != 0,
                (df_day_result["pred"] - df_day_result["real"]).abs() / df_day_result["real"] * 100,
                0
            )

            df_day_result["abs_error"] = (df_day_result["pred"] - df_day_result["real"]).abs()
            df_day_result['days_since_start'] = idx_day

            real_for_opt.append(df_day_result)

            # guardamos todos los precios predichos en el horizonte para hacer la predicción
            predictions_for_opt.append(pd.DataFrame({
                "datetime": X_day["datetime"],
                "pred": y_pred
            }))
        dict_price_pred_real[nombre_indicador] = pd.concat(real_for_opt, ignore_index=True)
        dict_optimization_input[nombre_indicador] = pd.concat(predictions_for_opt, ignore_index=True)

    return dict_optimization_input, dict_price_pred_real, dict_test_vs_pred

