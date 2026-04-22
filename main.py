# -*- coding: utf-8 -*-
"""
Hlavný spúšťací skript pre Diplomovú Prácu
Názov práce: Hybridný prístup strojového učenia a simulácie pri modelovaní finančných trhov
Autor: Dominika Melicherová

Tento skript orchestruje celý workflow:
1. Stiahnutie a načítanie dát
2. Predspracovanie a čistenie dát
3. Feature engineering
4. Trénovanie modelov (klasické ML aj deep learning)
5. Vyhodnotenie a porovnanie modelov
6. Simulácie (Monte Carlo, Backtesting)
7. Generovanie reportov a vizualizácií

Použitie:
    python main.py                    # Spustí kompletný pipeline
    python main.py --ticker AAPL      # Analýza konkrétnej akcie
    python main.py --skip-download    # Preskočí sťahovanie dát
    python main.py --quick            # Rýchly mód (menej modelov)
"""

import os
import sys
import argparse
import traceback
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Potlačenie varovných hlásení
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import vlastných modulov
import config
import data_downloader
import data_preprocessing
import feature_engineering
import visualization as viz
import models_classical
import model_evaluation
import simulation

# Kontrola dostupnosti TensorFlow
import models_deep_learning
DEEP_LEARNING_AVAILABLE = models_deep_learning.TENSORFLOW_AVAILABLE
if not DEEP_LEARNING_AVAILABLE:
    print("[VAROVANIE] Deep learning modely nie sú dostupné (TensorFlow nie je nainštalovaný)")


def print_header(text: str) -> None:
    """Vypíše formátovaný nadpis sekcie."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_subheader(text: str) -> None:
    """Vypíše formátovaný podnadpis."""
    print(f"\n--- {text} ---")


# =============================================================================
# FÁZA 1: SŤAHOVANIE A NAČÍTANIE DÁT
# =============================================================================

def phase_1_data_acquisition(
    tickers: list = None,
    skip_download: bool = False
) -> dict:
    """
    Fáza 1: Stiahnutie alebo načítanie dát.
    
    Args:
        tickers (list, optional): Zoznam tickerov
        skip_download (bool): Preskočiť sťahovanie a načítať lokálne dáta
        
    Returns:
        dict: Slovník s dátami pre každý ticker
    """
    print_header("FÁZA 1: SŤAHOVANIE A NAČÍTANIE DÁT")
    
    if tickers is None:
        tickers = config.get_ticker_list()
    
    all_data = {}
    
    for ticker in tickers:
        print_subheader(f"Spracovanie {ticker}")
        
        try:
            if skip_download:
                # Pokus o načítanie lokálnych dát
                try:
                    df = data_downloader.load_stock_data(ticker)
                except FileNotFoundError:
                    print(f"[INFO] Lokálne dáta pre {ticker} neexistujú, sťahujem...")
                    df = data_downloader.download_stock_data(ticker)
            else:
                df = data_downloader.download_stock_data(ticker)
            
            all_data[ticker] = df
            data_downloader.print_data_summary(df, ticker)
            
        except Exception as e:
            print(f"[CHYBA] Nepodarilo sa získať dáta pre {ticker}: {e}")
            continue
    
    print(f"\n[OK] Fáza 1 dokončená. Načítaných {len(all_data)} tickerov.")
    
    return all_data


# =============================================================================
# FÁZA 2: PREDSPRACOVANIE DÁT
# =============================================================================

def phase_2_preprocessing(data: dict) -> dict:
    """
    Fáza 2: Predspracovanie dát.
    
    Args:
        data (dict): Surové dáta
        
    Returns:
        dict: Predspracované dáta
    """
    print_header("FÁZA 2: PREDSPRACOVANIE DÁT")
    
    processed_data = {}
    
    for ticker, df in data.items():
        print_subheader(f"Predspracovanie {ticker}")
        
        try:
            df_processed = data_preprocessing.preprocess_pipeline(
                df, 
                ticker=ticker,
                save_to_file=True
            )
            processed_data[ticker] = df_processed
            
        except Exception as e:
            print(f"[CHYBA] Predspracovanie {ticker} zlyhalo: {e}")
            continue
    
    print(f"\n[OK] Fáza 2 dokončená. Predspracovaných {len(processed_data)} tickerov.")
    
    return processed_data


# =============================================================================
# FÁZA 3: FEATURE ENGINEERING
# =============================================================================

def phase_3_feature_engineering(data: dict) -> dict:
    """
    Fáza 3: Vytvorenie features.
    
    Args:
        data (dict): Predspracované dáta
        
    Returns:
        dict: Dáta s features
    """
    print_header("FÁZA 3: FEATURE ENGINEERING")
    
    feature_data = {}
    
    for ticker, df in data.items():
        print_subheader(f"Feature engineering pre {ticker}")
        
        try:
            df_features = feature_engineering.create_all_features(
                df,
                ticker=ticker,
                save_to_file=True
            )
            feature_data[ticker] = df_features
            
            # Súhrn features
            summary = feature_engineering.get_feature_importance_summary(df_features)
            print(f"\nPočet features podľa kategórií:")
            for category, features in summary.items():
                if features:
                    print(f"  {category}: {len(features)}\n     {features}")
            
        except Exception as e:
            print(f"[CHYBA] Feature engineering pre {ticker} zlyhal: {e}")
            continue
    
    print(f"\n[OK] Fáza 3 dokončená.")
    
    return feature_data


# =============================================================================
# FÁZA 4: VIZUALIZÁCIA DÁT
# =============================================================================

def phase_4_visualization(data: dict) -> None:
    """
    Fáza 4: Vytvorenie vizualizácií.
    
    Args:
        data (dict): Dáta s features
    """
    print_header("FÁZA 4: VIZUALIZÁCIA DÁT")
    
    for ticker, df in data.items():
        print_subheader(f"Vizualizácie pre {ticker}")
        
        try:
            viz.create_visualization_report(df, ticker)
        except Exception as e:
            print(f"[CHYBA] Vizualizácia pre {ticker} zlyhala: {e}")
            continue
    
    print(f"\n[OK] Fáza 4 dokončená. Grafy uložené do {config.PLOTS_DIR}")


# =============================================================================
# FÁZA 5: TRÉNOVANIE KLASICKÝCH ML MODELOV
# =============================================================================

def phase_5_classical_ml(
    data: dict,
    target_column: str = 'Target_Direction_252d',
    quick_mode: bool = False
) -> dict:
    """
    Fáza 5: Trénovanie klasických ML modelov.
    
    Args:
        data (dict): Dáta s features
        target_column (str): Cieľový stĺpec
        quick_mode (bool): Rýchly mód - menej modelov
        
    Returns:
        dict: Výsledky modelov pre každý ticker
    """
    print_header("FÁZA 5: TRÉNOVANIE KLASICKÝCH ML MODELOV")
    
    all_results = {}
    
    for ticker, df in data.items():
        print_subheader(f"Trénovanie modelov pre {ticker}")
        
        try:
            # Získanie feature stĺpcov
            feature_cols = feature_engineering.get_feature_list(df)
            
            # Rozdelenie dát
            split = data_preprocessing.split_data(
                df,
                target_column=target_column,
                feature_columns=feature_cols
            )
            
            X_train, y_train = split['X_train'], split['y_train']
            X_val, y_val = split['X_val'], split['y_val']
            X_test, y_test = split['X_test'], split['y_test']
            feature_names = split['feature_names']
            
            print(f"\nVeľkosť trénovacej sady: {len(X_train)}")
            print(f"Veľkosť validačnej sady: {len(X_val)}")
            print(f"Veľkosť testovacej sady: {len(X_test)}")
            print(f"Počet features: {len(feature_names)}")
            
            # Vytvorenie modelov
            if quick_mode:
                # V rýchlom móde len základné modely
                models = {
                    'Random Forest': models_classical.RandomForestModel('classification'),
                    'XGBoost': models_classical.XGBoostModel('classification') if models_classical.XGBOOST_AVAILABLE else None,
                    'Logistická regresia': models_classical.LogisticRegressionModel(),
                }
                models = {k: v for k, v in models.items() if v is not None}
            else:
                models = models_classical.create_all_classification_models()
            
            # Trénovanie a vyhodnotenie
            comparator = model_evaluation.ModelComparator('classification')
            
            for model_name, model in models.items():
                print(f"\n  Trénovanie: {model_name}")
                
                try:
                    # Build model ak ešte nie je
                    if not hasattr(model, 'model') or model.model is None:
                        model.build_model()
                    
                    # Trénovanie
                    model.train(X_train, y_train, feature_names)
                    
                    # Predikcia na testovacej sade
                    y_pred = model.predict(X_test)
                    
                    # Pravdepodobnosti ak sú dostupné
                    y_pred_proba = None
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_pred_proba = model.predict_proba(X_test)
                            if len(y_pred_proba.shape) > 1:
                                y_pred_proba = y_pred_proba[:, 1]
                        except Exception:
                            pass
                    
                    feature_importance = model.get_feature_importance()
                    if feature_importance is not None:
                        print(f"\nTop 5 najdôležitejších features:")
                        indices = np.argsort(feature_importance)[::-1][:5]
                        for i in indices:
                            print(f"  {feature_cols[i]}: {feature_importance[i]:.4f}")
                    # Pridanie do komparátora
                    comparator.add_result(model_name, y_test, y_pred, y_pred_proba)
                    
                    # Uloženie modelu
                    model.save_model()
                    
                except Exception as e:
                    print(f"    [CHYBA] {model_name}: {e}")
                    continue
            
            # Výpis porovnania
            comparator.print_comparison()
            
            # Uloženie výsledkov
            all_results[ticker] = {
                'comparator': comparator,
                'results': comparator.results,
                'best_model': comparator.find_best_model(),
                'X_test': X_test,
                'y_test': y_test,
                'feature_names': feature_names
            }
            
            # Generovanie reportu
            comparator.generate_report(
                os.path.join(config.RESULTS_DIR, f"{ticker}_classical_ml")
            )
            
        except Exception as e:
            print(f"[CHYBA] ML pre {ticker} zlyhalo: {e}")
            traceback.print_exc()
            continue
    
    print(f"\n[OK] Fáza 5 (klasifikácia) dokončená.")
    
    return all_results


# =============================================================================
# FÁZA 5b: TRÉNOVANIE KLASICKÝCH ML MODELOV (REGRESIA)
# =============================================================================

def phase_5b_classical_ml_regression(
    data: dict,
    target_column: str = 'Target_Return_252d',
    quick_mode: bool = False
) -> dict:
    """
    Fáza 5b: Trénovanie klasických ML regresných modelov.
    
    Args:
        data (dict): Dáta s features
        target_column (str): Cieľový stĺpec (regresný)
        quick_mode (bool): Rýchly mód - menej modelov
        
    Returns:
        dict: Výsledky modelov pre každý ticker
    """
    print_header("FÁZA 5b: TRÉNOVANIE KLASICKÝCH ML MODELOV (REGRESIA)")
    
    all_results = {}
    
    for ticker, df in data.items():
        print_subheader(f"Trénovanie regresných modelov pre {ticker}")
        
        try:
            feature_cols = feature_engineering.get_feature_list(df)
            
            split = data_preprocessing.split_data(
                df,
                target_column=target_column,
                feature_columns=feature_cols
            )
            
            X_train, y_train = split['X_train'], split['y_train']
            X_val, y_val = split['X_val'], split['y_val']
            X_test, y_test = split['X_test'], split['y_test']
            feature_names = split['feature_names']
            
            print(f"\nVeľkosť trénovacej sady: {len(X_train)}")
            print(f"Veľkosť validačnej sady: {len(X_val)}")
            print(f"Veľkosť testovacej sady: {len(X_test)}")
            print(f"Počet features: {len(feature_names)}")
            
            if quick_mode:
                models = {
                    'Random Forest': models_classical.RandomForestModel('regression'),
                    'XGBoost': models_classical.XGBoostModel('regression') if models_classical.XGBOOST_AVAILABLE else None,
                    'Ridge': models_classical.RidgeRegressionModel(),
                }
                models = {k: v for k, v in models.items() if v is not None}
            else:
                models = models_classical.create_all_regression_models()
            
            comparator = model_evaluation.ModelComparator('regression')
            
            for model_name, model in models.items():
                print(f"\n  Trénovanie: {model_name}")
                
                try:
                    if not hasattr(model, 'model') or model.model is None:
                        model.build_model()
                    
                    model.train(X_train, y_train, feature_names)
                    
                    y_pred = model.predict(X_test)
                    
                    comparator.add_result(model_name, y_test, y_pred)
                    
                    model.save_model()
                    
                except Exception as e:
                    print(f"    [CHYBA] {model_name}: {e}")
                    continue
            
            comparator.print_comparison()
            
            all_results[ticker] = {
                'comparator': comparator,
                'results': comparator.results,
                'best_model': comparator.find_best_model(),
                'X_test': X_test,
                'y_test': y_test,
                'feature_names': feature_names
            }
            
            comparator.generate_report(
                os.path.join(config.RESULTS_DIR, f"{ticker}_classical_ml_regression")
            )
            
        except Exception as e:
            print(f"[CHYBA] ML regresia pre {ticker} zlyhala: {e}")
            traceback.print_exc()
            continue
    
    print(f"\n[OK] Fáza 5b (regresia) dokončená.")
    
    return all_results


# =============================================================================
# FÁZA 6: TRÉNOVANIE DEEP LEARNING MODELOV
# =============================================================================

def phase_6_deep_learning(
    data: dict,
    quick_mode: bool = False
) -> dict:
    """
    Fáza 6: Trénovanie deep learning modelov (klasifikácia aj regresia).
    
    Používa create_all_deep_learning_models() z models_deep_learning na
    vytvorenie všetkých modelov (LSTM, GRU, CNN-LSTM, MLP) pre obe úlohy.
    V rýchlom režime sa vytvoria len Simple LSTM a Simple GRU.
    
    Args:
        data (dict): Dáta s features
        quick_mode (bool): Rýchly mód
        
    Returns:
        dict: Výsledky deep learning modelov
              Štruktúra: {ticker: {'classification': {...}, 'regression': {...}}}
    """
    if not DEEP_LEARNING_AVAILABLE:
        print_header("FÁZA 6: DEEP LEARNING (PRESKOČENÉ)")
        print("[INFO] TensorFlow nie je nainštalovaný. Deep learning preskočený.")
        return {}
    
    print_header("FÁZA 6: TRÉNOVANIE DEEP LEARNING MODELOV")
    
    sequence_length = config.LSTM_PARAMS['sequence_length']
    epochs = 10 if quick_mode else 50
    all_results = {}
    
    tasks = [
        ('classification', 'Target_Direction_252d'),
        ('regression', 'Target_Return_252d'),
    ]
    
    for ticker, df in data.items():
        print_subheader(f"Deep Learning pre {ticker}")
        
        ticker_results = {}
        
        for task, target_column in tasks:
            print(f"\n  --- DL {task.upper()} ---")
            
            try:
                feature_cols = feature_engineering.get_feature_list(df)
                
                split = data_preprocessing.split_data(
                    df,
                    target_column=target_column,
                    feature_columns=feature_cols
                )
                
                X_train, y_train = split['X_train'], split['y_train']
                X_val, y_val = split['X_val'], split['y_val']
                X_test, y_test = split['X_test'], split['y_test']
                
                # Príprava sekvencií (dáta sú už škálované cez split_data)
                X_train_seq, y_train_seq = models_deep_learning.prepare_sequences(
                    X_train, y_train, sequence_length
                )
                X_val_seq, y_val_seq = models_deep_learning.prepare_sequences(
                    X_val, y_val, sequence_length
                )
                X_test_seq, y_test_seq = models_deep_learning.prepare_sequences(
                    X_test, y_test, sequence_length
                )
                
                print(f"  Tvar sekvenčných dát: {X_train_seq.shape}")
                
                input_shape = (sequence_length, X_train.shape[1])
                input_dim = X_train.shape[1]
                
                # Vytvorenie všetkých modelov cez helper funkciu
                dl_models = models_deep_learning.create_all_deep_learning_models(
                    input_shape, task,
                    input_dim=input_dim,
                    quick=quick_mode
                )
                
                comparator = model_evaluation.ModelComparator(task)
                
                for model_name, model in dl_models.items():
                    print(f"\n  Trénovanie: {model_name}")
                    try:
                        if model.is_sequential:
                            models_deep_learning.train_deep_model(
                                model, X_train_seq, y_train_seq,
                                X_val_seq, y_val_seq, epochs=epochs
                            )
                            y_pred = model.predict(X_test_seq)
                            y_true = y_test_seq
                        else:
                            models_deep_learning.train_deep_model(
                                model, X_train, y_train,
                                X_val, y_val, epochs=epochs
                            )
                            y_pred = model.predict(X_test)
                            y_true = y_test
                        
                        if task == 'classification':
                            X_prob = X_test_seq if model.is_sequential else X_test
                            y_pred_proba = model.predict_proba(X_prob)
                            comparator.add_result(model_name, y_true, y_pred, y_pred_proba)
                        else:
                            comparator.add_result(model_name, y_true, y_pred)
                        
                        model.save_model()
                    except Exception as e:
                        print(f"    [CHYBA] {model_name}: {e}")
                        traceback.print_exc()
                
                if comparator.results:
                    comparator.print_comparison()
                    
                    ticker_results[task] = {
                        'comparator': comparator,
                        'results': comparator.results,
                        'best_model': comparator.find_best_model()
                    }
                    
                    comparator.generate_report(
                        os.path.join(config.RESULTS_DIR, f"{ticker}_dl_{task}")
                    )
                
            except Exception as e:
                print(f"[CHYBA] DL {task} pre {ticker} zlyhalo: {e}")
                traceback.print_exc()
                continue
        
        if ticker_results:
            all_results[ticker] = ticker_results
    
    print(f"\n[OK] Fáza 6 dokončená.")
    
    return all_results


# =============================================================================
# FÁZA 7: SIMULÁCIE
# =============================================================================

def phase_7_simulations(
    data: dict,
    ml_results: dict = None
) -> dict:
    """
    Fáza 7: Hybridný prístup - fitovanie rozdelenia, MC simulácie, backtesting.

    Postup pre každý ticker:
      1. Identifikácia rozdelenia log-výnosov (fitovanie viacerých kandidátov,
         výber najlepšieho podľa AIC / KS testu).
      2. Generovanie simulovaných dát z nafitovaného rozdelenia.
      3. Porovnanie GBM (normálne rozdelenie) vs. nafitovaného rozdelenia.
      4. Backtesting obchodných stratégií.

    Args:
        data (dict): Dáta s features
        ml_results (dict, optional): Výsledky ML modelov (rezervované pre budúce
            prepojenie ML drift-predikcií so simuláciou)

    Returns:
        dict: Výsledky simulácií pre každý ticker
    """
    print_header("FÁZA 7: HYBRIDNÉ SIMULÁCIE (FITOVANIE ROZDELENIA + MONTE CARLO)")

    all_simulations = {}

    for ticker, df in data.items():
        print_subheader(f"Simulácie pre {ticker}")

        prices = df['Close'].values
        plot_dir = os.path.join(config.PLOTS_DIR, ticker)
        os.makedirs(plot_dir, exist_ok=True)

        try:
            # ------------------------------------------------------------------
            # KROK 1: Identifikácia rozdelenia výnosov
            # ------------------------------------------------------------------
            print("\n  Krok 1: Fitovanie rozdelenia log-výnosov...")
            log_returns = np.diff(np.log(prices))

            fitter = simulation.ReturnDistributionFitter(log_returns)
            fitter.fit()
            fitter.print_moments_report()

            best_dist_name = fitter.all_results[fitter.best_distribution]['name_sk']
            print(f"\n  -> Identifikované najlepšie rozdelenie: {best_dist_name}")

            # Graf fitovaného rozdelenia
            viz.plot_distribution_fit(
                log_returns, fitter, ticker,
                save_path=os.path.join(plot_dir, "distribution_fit.png")
            )
            plt.close('all')

            # ------------------------------------------------------------------
            # KROK 2 & 3: GBM simulácia + simulácia s nafitovaným rozdelením
            # ------------------------------------------------------------------
            print("\n  Krok 2: GBM Monte Carlo simulácia (referencia)...")
            mc_gbm = simulation.MonteCarloSimulator(
                prices,
                n_simulations=config.MONTE_CARLO_SIMULATIONS,
                n_days=config.MONTE_CARLO_DAYS
            )
            gbm_sims = mc_gbm.simulate_gbm(seed=config.RANDOM_SEED)
            mc_gbm.print_report()

            print("\n  Krok 3: MC simulácia s nafitovaným rozdelením (hybridný prístup)...")
            mc_fitted = simulation.MonteCarloSimulator(
                prices,
                n_simulations=config.MONTE_CARLO_SIMULATIONS,
                n_days=config.MONTE_CARLO_DAYS
            )
            fitted_sims = mc_fitted.simulate_with_fitted_distribution(
                fitter, seed=config.RANDOM_SEED
            )
            mc_fitted.print_report()

            # Graf GBM simulácie (pôvodný)
            viz.plot_monte_carlo_simulation(
                gbm_sims, prices[-1], ticker,
                save_path=os.path.join(plot_dir, "mc_gbm.png")
            )
            plt.close('all')

            # Graf nafitovanej simulácie
            viz.plot_monte_carlo_simulation(
                fitted_sims, prices[-1], ticker,
                save_path=os.path.join(plot_dir, "mc_fitted.png")
            )
            plt.close('all')

            # Porovnávací graf GBM vs. nafitované rozdelenie
            viz.plot_mc_comparison(
                gbm_sims, fitted_sims, prices[-1],
                dist_name=fitter.CANDIDATE_DISTRIBUTIONS[fitter.best_distribution],
                ticker=ticker,
                save_path=os.path.join(plot_dir, "mc_comparison.png")
            )
            plt.close('all')

            # Graf simulovaných výnosov a konečných cien (GBM)
            viz.plot_simulation_returns_and_prices(
                gbm_sims, prices[-1], ticker=f"{ticker} (GBM)",
                save_path=os.path.join(plot_dir, "mc_gbm_returns_prices.png")
            )
            plt.close('all')

            # Graf simulovaných výnosov a konečných cien (nafitované)
            viz.plot_simulation_returns_and_prices(
                fitted_sims, prices[-1], ticker=f"{ticker} ({best_dist_name})",
                save_path=os.path.join(plot_dir, "mc_fitted_returns_prices.png")
            )
            plt.close('all')

            # Sanity check simulácií
            print("\n  Sanity check simulácií...")
            sanity_gbm = mc_gbm.sanity_check()
            sanity_fitted = mc_fitted.sanity_check()

            # ------------------------------------------------------------------
            # KROK 4: Backtesting obchodných stratégií
            # ------------------------------------------------------------------
            print("\n  Krok 4: Backtesting obchodných stratégií...")

            # MA crossover stratégia
            print("    Backtesting MA stratégie...")
            ma_signals = simulation.moving_average_crossover_strategy(prices)
            bt_ma = simulation.Backtester()
            results_ma = bt_ma.run(prices, ma_signals)
            bt_ma.print_report()

            viz.plot_backtest_results(
                bt_ma.get_results_for_visualization(),
                save_path=os.path.join(plot_dir, "backtest_ma.png")
            )
            plt.close('all')

            # RSI stratégia
            print("    Backtesting RSI stratégie...")
            rsi_signals = simulation.rsi_strategy(prices)
            bt_rsi = simulation.Backtester()
            results_rsi = bt_rsi.run(prices, rsi_signals)
            bt_rsi.print_report()

            viz.plot_backtest_results(
                bt_rsi.get_results_for_visualization(),
                save_path=os.path.join(plot_dir, "backtest_rsi.png")
            )
            plt.close('all')

            # ------------------------------------------------------------------
            # Uloženie výsledkov
            # ------------------------------------------------------------------
            all_simulations[ticker] = {
                'distribution_fit': {
                    'best_distribution':      fitter.best_distribution,
                    'best_distribution_name': best_dist_name,
                    'best_params':            fitter.best_params,
                    'all_results':            {
                        k: {kk: vv for kk, vv in v.items() if kk != 'params'}
                        for k, v in fitter.all_results.items()
                    },
                    'empirical_moments':      fitter.get_empirical_moments(),
                },
                'mc_gbm': {
                    'statistics': mc_gbm.calculate_statistics(),
                    'var_95':     mc_gbm.calculate_var(0.95),
                    'var_99':     mc_gbm.calculate_var(0.99),
                    'sanity_check': sanity_gbm,
                },
                'mc_fitted': {
                    'statistics': mc_fitted.calculate_statistics(),
                    'var_95':     mc_fitted.calculate_var(0.95),
                    'var_99':     mc_fitted.calculate_var(0.99),
                    'sanity_check': sanity_fitted,
                },
                'backtest_ma':  results_ma,
                'backtest_rsi': results_rsi,
            }

        except Exception as e:
            print(f"[CHYBA] Simulácie pre {ticker} zlyhali: {e}")
            traceback.print_exc()
            continue

    print(f"\n[OK] Fáza 7 dokončená.")

    return all_simulations


# =============================================================================
# FÁZA 8: ZÁVEREČNÝ REPORT
# =============================================================================

def phase_8_final_report(
    ml_results: dict,
    ml_regression_results: dict,
    dl_results: dict,
    simulation_results: dict
) -> None:
    """
    Fáza 8: Generovanie kompletného záverečného reportu.

    Report obsahuje:
      - Konfiguračný súhrn
      - Klasické ML klasifikačné výsledky
      - Klasické ML regresné výsledky
      - Deep Learning výsledky
      - Výsledky fitovaných rozdelení výnosov
      - Monte Carlo simulácie (GBM vs. nafitované) vrátane sanity checkov
      - Backtesting výsledky
      - Celkový súhrn s najlepšími modelmi a kľúčovými zisteniami

    Args:
        ml_results (dict): Výsledky klasických ML klasifikačných modelov
        ml_regression_results (dict): Výsledky klasických ML regresných modelov
        dl_results (dict): Výsledky deep learning modelov
        simulation_results (dict): Výsledky simulácií
    """
    print_header("FÁZA 8: ZÁVEREČNÝ REPORT")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(config.RESULTS_DIR, f"final_report_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, "report.txt")
    W = 80  # šírka riadku

    def sep(char='='):
        return char * W + "\n"

    def heading(title):
        return "\n" + sep() + f"  {title}\n" + sep()

    def subheading(title):
        return "\n" + sep('-') + f"  {title}\n" + sep('-')

    with open(report_path, 'w', encoding='utf-8') as f:
        # ── Hlavička ──
        f.write(sep())
        f.write("  ZÁVEREČNÝ REPORT\n")
        f.write("  Strojové učenie a simulácia pri modelovaní finančných trhov\n")
        f.write(f"  Vygenerované: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(sep())

        # ── 1. Konfigurácia ──
        f.write(heading("1. KONFIGURÁCIA"))
        f.write(f"  Tickery       : {', '.join(config.get_ticker_list())}\n")
        f.write(f"  Obdobie       : {config.START_DATE} – {config.END_DATE}\n")
        f.write(f"  Train / Val / Test : "
                f"{config.TRAIN_RATIO} / {config.VAL_RATIO} / {config.TEST_RATIO}\n")
        f.write(f"  Random seed   : {config.RANDOM_SEED}\n")
        f.write(f"  MC simulácií  : {config.MONTE_CARLO_SIMULATIONS}\n")
        f.write(f"  MC horizont   : {config.MONTE_CARLO_DAYS} dní\n")
        f.write(f"  Drift metóda  : {config.SIMULATION_DRIFT_METHOD}\n")
        if config.SIMULATION_DRIFT_METHOD == 'custom':
            f.write(f"  Očak. ročný výnos: {config.SIMULATION_EXPECTED_ANNUAL_RETURN:.1%}\n")
        elif config.SIMULATION_DRIFT_METHOD == 'risk_neutral':
            f.write(f"  Bezriziková miera: {config.RISK_FREE_RATE:.1%}\n")

        # ── 2. Klasické ML modely (klasifikácia) ──
        f.write(heading("2. KLASICKÉ ML MODELY (KLASIFIKÁCIA)"))
        best_models_summary = []

        for ticker, results in ml_results.items():
            f.write(subheading(f"Ticker: {ticker}"))
            best_model, best_score = results.get('best_model', (None, None))
            if best_model:
                f.write(f"  >>> Najlepší model: {best_model} "
                        f"(accuracy: {best_score:.4f}) <<<\n\n")
                best_models_summary.append((ticker, 'ML-clf', best_model, best_score))

            if 'results' in results:
                f.write(f"  {'Model':<28} {'Accuracy':>9} {'Precision':>10} "
                        f"{'Recall':>8} {'F1':>8}\n")
                f.write(f"  {'-'*66}\n")
                for model_name, metrics in results['results'].items():
                    acc = metrics.get('accuracy', 0)
                    prec = metrics.get('precision', 0)
                    rec = metrics.get('recall', 0)
                    f1 = metrics.get('f1_score', 0)
                    f.write(f"  {model_name:<28} {acc:>9.4f} {prec:>10.4f} "
                            f"{rec:>8.4f} {f1:>8.4f}\n")

        # ── 3. Klasické ML modely (regresia) ──
        f.write(heading("3. KLASICKÉ ML MODELY (REGRESIA)"))
        if ml_regression_results:
            for ticker, results in ml_regression_results.items():
                f.write(subheading(f"Ticker: {ticker}"))
                best_model, best_score = results.get('best_model', (None, None))
                if best_model:
                    f.write(f"  >>> Najlepší model: {best_model} "
                            f"(RMSE: {best_score:.4f}) <<<\n\n")
                    best_models_summary.append((ticker, 'ML-reg', best_model, best_score))

                if 'results' in results:
                    f.write(f"  {'Model':<28} {'RMSE':>9} {'MAE':>9} "
                            f"{'R²':>9} {'MAPE':>9}\n")
                    f.write(f"  {'-'*68}\n")
                    for model_name, metrics in results['results'].items():
                        rmse = metrics.get('rmse', 0)
                        mae = metrics.get('mae', 0)
                        r2 = metrics.get('r2', 0)
                        mape = metrics.get('mape', 0) or 0
                        f.write(f"  {model_name:<28} {rmse:>9.4f} {mae:>9.4f} "
                                f"{r2:>9.4f} {mape:>8.2f}%\n")
        else:
            f.write("  Regresná fáza nebola spustená alebo nevyprodukovala výsledky.\n")

        # ── 4. Deep Learning modely ──
        f.write(heading("4. DEEP LEARNING MODELY"))
        if dl_results:
            # 4a. Klasifikácia
            f.write(subheading("4a. Deep Learning – Klasifikácia"))
            has_clf = False
            for ticker, ticker_res in dl_results.items():
                clf_res = ticker_res.get('classification')
                if not clf_res:
                    continue
                has_clf = True
                f.write(f"\n  Ticker: {ticker}\n")
                best = clf_res.get('best_model')
                if best:
                    bname, bscore = best
                    f.write(f"  >>> Najlepší DL model: {bname} "
                            f"(accuracy: {bscore:.4f}) <<<\n\n")
                    best_models_summary.append((ticker, 'DL-clf', bname, bscore))
                if 'results' in clf_res:
                    f.write(f"  {'Model':<28} {'Accuracy':>9} {'Precision':>10} "
                            f"{'Recall':>8} {'F1':>8}\n")
                    f.write(f"  {'-'*66}\n")
                    for model_name, metrics in clf_res['results'].items():
                        acc = metrics.get('accuracy', 0)
                        prec = metrics.get('precision', 0)
                        rec = metrics.get('recall', 0)
                        f1 = metrics.get('f1_score', 0)
                        f.write(f"  {model_name:<28} {acc:>9.4f} {prec:>10.4f} "
                                f"{rec:>8.4f} {f1:>8.4f}\n")
            if not has_clf:
                f.write("  DL klasifikácia nebola spustená alebo nevyprodukovala výsledky.\n")

            # 4b. Regresia
            f.write(subheading("4b. Deep Learning – Regresia"))
            has_reg = False
            for ticker, ticker_res in dl_results.items():
                reg_res = ticker_res.get('regression')
                if not reg_res:
                    continue
                has_reg = True
                f.write(f"\n  Ticker: {ticker}\n")
                best = reg_res.get('best_model')
                if best:
                    bname, bscore = best
                    f.write(f"  >>> Najlepší DL model: {bname} "
                            f"(RMSE: {bscore:.4f}) <<<\n\n")
                    best_models_summary.append((ticker, 'DL-reg', bname, bscore))
                if 'results' in reg_res:
                    f.write(f"  {'Model':<28} {'RMSE':>9} {'MAE':>9} "
                            f"{'R²':>9} {'MAPE':>9}\n")
                    f.write(f"  {'-'*68}\n")
                    for model_name, metrics in reg_res['results'].items():
                        rmse = metrics.get('rmse', 0)
                        mae = metrics.get('mae', 0)
                        r2 = metrics.get('r2', 0)
                        mape = metrics.get('mape', 0) or 0
                        f.write(f"  {model_name:<28} {rmse:>9.4f} {mae:>9.4f} "
                                f"{r2:>9.4f} {mape:>8.2f}%\n")
            if not has_reg:
                f.write("  DL regresia nebola spustená alebo nevyprodukovala výsledky.\n")
        else:
            f.write("  Deep Learning fáza nebola spustená alebo nevyprodukovala výsledky.\n")

        # ── 5. Hybridné simulácie ──
        f.write(heading("5. HYBRIDNÉ SIMULÁCIE (FITOVANIE ROZDELENIA + MONTE CARLO)"))

        for ticker, results in simulation_results.items():
            f.write(subheading(f"Ticker: {ticker}"))

            # 5a. Fitovanie rozdelenia
            dist_fit = results.get('distribution_fit', {})
            if dist_fit:
                f.write("  5a. Fitovanie rozdelenia log-výnosov\n")
                f.write(f"      Najlepšie rozdelenie: "
                        f"{dist_fit.get('best_distribution_name', 'N/A')} "
                        f"({dist_fit.get('best_distribution', '')})\n")
                moments = dist_fit.get('empirical_moments', {})
                f.write(f"      Priemer (μ)           : {moments.get('mean', 0):.6f}\n")
                f.write(f"      Volatilita (σ)        : {moments.get('std', 0):.6f}\n")
                f.write(f"      Šikmosť               : {moments.get('skewness', 0):.4f}\n")
                f.write(f"      Prebytočná špicatosť  : {moments.get('kurtosis', 0):.4f}\n")
                all_fits = dist_fit.get('all_results', {})
                if all_fits:
                    f.write(f"\n      {'Rozdelenie':<16} {'AIC':>10} {'KS p-val':>10}\n")
                    f.write(f"      {'-'*38}\n")
                    for dname, dres in sorted(all_fits.items(),
                                              key=lambda x: x[1].get('aic', 9999)):
                        marker = " *" if dname == dist_fit.get('best_distribution') else ""
                        f.write(f"      {dname:<16} {dres.get('aic', 0):>10.2f} "
                                f"{dres.get('ks_pvalue', 0):>10.4f}{marker}\n")

            # 5b/5c. GBM vs. Nafitovaná simulácia
            for mc_key, mc_label in [('mc_gbm', 'GBM (normálne rozdelenie)'),
                                      ('mc_fitted', 'Nafitované rozdelenie')]:
                mc = results.get(mc_key, {})
                if not mc:
                    continue
                stats = mc.get('statistics', {})
                var95 = mc.get('var_95', {})
                var99 = mc.get('var_99', {})
                sanity = mc.get('sanity_check', {})

                f.write(f"\n  Monte Carlo – {mc_label}\n")
                f.write(f"      Priemerná konečná cena : {stats.get('priemerna_konecna_cena', 0):.2f}\n")
                f.write(f"      Medián konečnej ceny   : {stats.get('medianová_konecna_cena', 0):.2f}\n")
                f.write(f"      5. / 95. percentil     : {stats.get('percentil_5', 0):.2f} / "
                        f"{stats.get('percentil_95', 0):.2f}\n")
                f.write(f"      Priemerný výnos        : {stats.get('priemerny_vynos', 0):.2f} %\n")
                f.write(f"      P(rast)                : {stats.get('pravdepodobnost_rastu', 0):.1f} %\n")
                f.write(f"      VaR 95 % / 99 %       : {var95.get('var', 0):.2f} % / "
                        f"{var99.get('var', 0):.2f} %\n")
                f.write(f"      CVaR 95 % / 99 %      : {var95.get('cvar', 0):.2f} % / "
                        f"{var99.get('cvar', 0):.2f} %\n")

                # Sanity check výsledky
                if sanity:
                    is_ok = sanity.get('is_realistic', True)
                    status = "PASSED" if is_ok else "VAROVANIA"
                    f.write(f"      Sanity check           : {status}\n")
                    for w in sanity.get('warnings', []):
                        f.write(f"        ⚠ {w}\n")

            # 5d. Backtesting
            f.write("\n  Backtesting stratégií\n")
            f.write(f"      {'Stratégia':<12} {'Výnos':>10} {'Sharpe':>9} "
                    f"{'Max DD':>9} {'Win Rate':>10} {'vs B&H':>10}\n")
            f.write(f"      {'-'*62}\n")
            for bt_key, bt_label in [('backtest_ma', 'MA Cross'),
                                      ('backtest_rsi', 'RSI')]:
                bt = results.get(bt_key, {})
                if bt:
                    f.write(f"      {bt_label:<12} "
                            f"{bt.get('celkovy_vynos_pct', 0):>9.2f}% "
                            f"{bt.get('sharpe_ratio', 0):>9.4f} "
                            f"{bt.get('max_drawdown_pct', 0):>8.2f}% "
                            f"{bt.get('win_rate_pct', 0):>9.1f}% "
                            f"{bt.get('prekonanie_buy_hold_pct', 0):>9.2f}%\n")

        # ── 6. Celkový súhrn ──
        f.write(heading("6. CELKOVÝ SÚHRN"))

        if best_models_summary:
            f.write("  Najlepšie modely pre každý ticker:\n\n")
            f.write(f"  {'Ticker':<8} {'Typ':<8} {'Model':<28} {'Metrika':>9}\n")
            f.write(f"  {'-'*57}\n")
            for ticker, mtype, mname, mscore in best_models_summary:
                f.write(f"  {ticker:<8} {mtype:<8} {mname:<28} {mscore:>9.4f}\n")

        # Kľúčové zistenia zo simulácií
        if simulation_results:
            f.write("\n  Kľúčové zistenia zo simulácií:\n")
            for ticker, results in simulation_results.items():
                dist_fit = results.get('distribution_fit', {})
                mc_fit = results.get('mc_fitted', {})
                mc_gbm_r = results.get('mc_gbm', {})
                stats_fit = mc_fit.get('statistics', {})
                stats_gbm = mc_gbm_r.get('statistics', {})

                f.write(f"\n    {ticker}:\n")
                f.write(f"      Identifikované rozdelenie : "
                        f"{dist_fit.get('best_distribution_name', 'N/A')}\n")
                if stats_fit and stats_gbm:
                    diff = (stats_fit.get('priemerny_vynos', 0)
                            - stats_gbm.get('priemerny_vynos', 0))
                    f.write(f"      Rozdiel priem. výnosu (nafitované - GBM): "
                            f"{diff:+.2f} p.p.\n")
                    f.write(f"      P(rast) nafitované / GBM : "
                            f"{stats_fit.get('pravdepodobnost_rastu', 0):.1f} % / "
                            f"{stats_gbm.get('pravdepodobnost_rastu', 0):.1f} %\n")

                # Sanity summary
                for mc_key, mc_label in [('mc_gbm', 'GBM'), ('mc_fitted', 'Nafitované')]:
                    sc = results.get(mc_key, {}).get('sanity_check', {})
                    if sc and not sc.get('is_realistic', True):
                        f.write(f"      ⚠ {mc_label} simulácia má "
                                f"{sc.get('n_warnings', 0)} varovanie(í)\n")

        f.write("\n" + sep())
        f.write(f"  Grafy uložené v   : {config.PLOTS_DIR}\n")
        f.write(f"  Modely uložené v  : {config.MODELS_DIR}\n")
        f.write(f"  Výsledky uložené v: {config.RESULTS_DIR}\n")
        f.write(sep())
        f.write("  Koniec reportu\n")
        f.write(sep())

    # ── Konzolový súhrn ──
    print(f"\n[OK] Záverečný report uložený do {report_path}")

    print("\n" + "=" * W)
    print("  SÚHRN VÝSLEDKOV")
    print("=" * W)
    if best_models_summary:
        for ticker, mtype, mname, mscore in best_models_summary:
            metric_label = "RMSE" if mtype in ('ML-reg', 'DL-reg') else "accuracy"
            print(f"  {ticker:<8} {mtype:<8} {mname:<28} {metric_label}={mscore:.4f}")
    for ticker, results in simulation_results.items():
        dist_name = results.get('distribution_fit', {}).get('best_distribution_name', '?')
        mc_fit = results.get('mc_fitted', {}).get('statistics', {})
        ret = mc_fit.get('priemerny_vynos', 0)
        prob = mc_fit.get('pravdepodobnost_rastu', 0)
        print(f"  {ticker:<8} Rozdelenie: {dist_name:<24} "
              f"MC výnos: {ret:+.2f}%, P(rast): {prob:.1f}%")
    print("=" * W)


# =============================================================================
# HLAVNÁ FUNKCIA
# =============================================================================

def main():
    """Hlavná funkcia spúšťajúca celý pipeline."""
    
    # Parsovanie argumentov
    parser = argparse.ArgumentParser(
        description='Strojové učenie a simulácia na finančných trhoch'
    )
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        default=None,
        help='Konkrétny ticker na analýzu (napr. AAPL)'
    )
    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        default=None,
        help='Zoznam tickerov na analýzu'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Preskočiť sťahovanie dát (použiť lokálne)'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Rýchly mód (menej modelov, kratšie trénovanie)'
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Preskočiť vytváranie vizualizácií'
    )
    parser.add_argument(
        '--skip-dl',
        action='store_true',
        help='Preskočiť deep learning modely'
    )
    parser.add_argument(
        '--skip-simulation',
        action='store_true',
        help='Preskočiť simulácie'
    )
    
    args = parser.parse_args()
    
    # Určenie tickerov
    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = args.tickers
    else:
        tickers = config.get_ticker_list()
    
    # Úvodná správa
    print("\n" + "=" * 70)
    print("  STROJOVÉ UČENIE A SIMULÁCIA PRI MODELOVANÍ FINANČNÝCH TRHOV")
    print("  Diplomová práca - Hybridný prístup")
    print("=" * 70)
    print(f"\nDátum spustenia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tickery na analýzu: {', '.join(tickers)}")
    print(f"Rýchly mód: {'Áno' if args.quick else 'Nie'}")
    config.print_config()
    
    # Spustenie pipeline
    try:
        # Fáza 1: Sťahovanie dát
        raw_data = phase_1_data_acquisition(tickers, args.skip_download)
        
        if not raw_data:
            print("[CHYBA] Žiadne dáta neboli načítané. Ukončujem.")
            return
        
        # Fáza 2: Predspracovanie
        processed_data = phase_2_preprocessing(raw_data)
        
        # Fáza 3: Feature engineering
        feature_data = phase_3_feature_engineering(processed_data)
        
        # Fáza 4: Vizualizácia
        if not args.skip_visualization:
            phase_4_visualization(feature_data)
        
        # Fáza 5: Klasické ML (klasifikácia)
        ml_results = phase_5_classical_ml(feature_data, quick_mode=args.quick)
        
        # Fáza 5b: Klasické ML (regresia)
        ml_regression_results = phase_5b_classical_ml_regression(
            feature_data, quick_mode=args.quick
        )
        
        # Fáza 6: Deep Learning
        dl_results = {}
        if not args.skip_dl:
            dl_results = phase_6_deep_learning(feature_data, quick_mode=args.quick)
        
        # Fáza 7: Simulácie
        simulation_results = {}
        if not args.skip_simulation:
            simulation_results = phase_7_simulations(feature_data, ml_results)
        
        # Fáza 8: Záverečný report
        phase_8_final_report(
            ml_results, ml_regression_results, dl_results, simulation_results
        )
        
        print("\n" + "=" * 70)
        print("  ANALÝZA ÚSPEŠNE DOKONČENÁ")
        print("=" * 70)
        print(f"\nVýsledky uložené v: {config.RESULTS_DIR}")
        print(f"Grafy uložené v: {config.PLOTS_DIR}")
        print(f"Modely uložené v: {config.MODELS_DIR}")
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Analýza prerušená používateľom.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[KRITICKÁ CHYBA] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
