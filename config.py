# -*- coding: utf-8 -*-
"""
Konfiguračný modul pre projekt strojového učenia na finančných trhoch.

Tento modul obsahuje všetky konfigurovateľné parametre projektu vrátane:
- Zoznam akciových tickerov na analýzu
- Časové obdobie pre sťahovanie dát
- Parametre pre technické indikátory
- Hyperparametre pre modely strojového učenia
- Cesty k súborom a adresárom

Autor: Diplomová práca - Strojové učenie a simulácia pri modelovaní finančných trhov
"""

import os
from datetime import datetime, timedelta

# =============================================================================
# ZÁKLADNÉ NASTAVENIA PROJEKTU
# =============================================================================

# Koreňový adresár projektu
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Adresáre pre ukladanie dát a výstupov
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# Vytvorenie adresárov ak neexistujú
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# =============================================================================
# KONFIGURÁCIA AKCIOVÝCH TICKEROV
# =============================================================================

# Zoznam akciových tickerov na analýzu
# Môžete ľahko upraviť tento zoznam podľa potreby
STOCK_TICKERS = [
    "AAPL",   # Apple Inc.
    "^GSPC",  # S&P 500
    "MSFT",   # Microsoft Corporation
#    "GOOGL",  # Alphabet Inc. (Google)
#    "AMZN",   # Amazon.com Inc.
#    "META",   # Meta Platforms Inc. (Facebook)
#    "TSLA",   # Tesla Inc.
#    "NVDA",   # NVIDIA Corporation
    "JPM",    # JPMorgan Chase & Co.
    "V",      # Visa Inc.
    "JNJ",    # Johnson & Johnson
]

# Predvolený ticker pre detailnú analýzu
DEFAULT_TICKER = "AAPL"

# =============================================================================
# KONFIGURÁCIA ČASOVÉHO OBDOBIA
# =============================================================================

# Počiatočný a koncový dátum pre sťahovanie dát
# Predvolené: posledných 5 rokov
END_DATE = "2025-01-01" #datetime.now().strftime("%Y-%m-%d")
START_DATE = "2010-01-01" #(datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")

# Alternatívne môžete nastaviť konkrétne dátumy:
# START_DATE = "2019-01-01"
# END_DATE = "2024-01-01"

# =============================================================================
# KONFIGURÁCIA TECHNICKÝCH INDIKÁTOROV
# =============================================================================

# Parametre pre kĺzavé priemery
SMA_PERIODS = [5, 10, 20, 50, 200]  # Jednoduché kĺzavé priemery
EMA_PERIODS = [12, 26, 50]          # Exponenciálne kĺzavé priemery

# Parametre pre RSI (Relative Strength Index)
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70  # Hranica prekúpenosti
RSI_OVERSOLD = 30    # Hranica prepredanosti

# Parametre pre MACD (Moving Average Convergence Divergence)
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# Parametre pre Bollinger Bands
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2  # Počet štandardných odchýlok

# Parametre pre ATR (Average True Range)
ATR_PERIOD = 14

# Parametre pre Stochastic Oscillator
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3

# Počet oneskorených (lagged) hodnôt pre features
LAG_PERIODS = [1, 2, 3, 5, 10]

# =============================================================================
# KONFIGURÁCIA PREDIKČNÝCH CIEĽOV
# =============================================================================

# Horizonty predikcie (v dňoch)
PREDICTION_HORIZONS = [1, 5, 10, 20, 252]  # 1 deň, 1 týždeň, 2 týždne, 1 mesiac, 1 rok

# Prah pre klasifikáciu smeru (ak je zmena väčšia ako tento prah, považuje sa za signifikantný pohyb)
DIRECTION_THRESHOLD = 2.0  # 0% = akýkoľvek pohyb

# =============================================================================
# KONFIGURÁCIA ROZDELENIA DÁT
# =============================================================================

# Pomer rozdelenia dát na trénovaciu, validačnú a testovaciu množinu
TRAIN_RATIO = 0.7   # 70% pre trénovanie
VAL_RATIO = 0.15    # 15% pre validáciu
TEST_RATIO = 0.15   # 15% pre testovanie

# Random Seed pre reprodukovateľnosť
RANDOM_SEED = 69

# =============================================================================
# KONFIGURÁCIA KLASICKÝCH ML MODELOV
# =============================================================================

# Random Forest
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_SEED,
    "n_jobs": -1  # Použiť všetky jadrá procesora
}

# XGBoost
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

# Support Vector Machine
SVM_PARAMS = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "random_state": RANDOM_SEED
}

# K-Nearest Neighbors
KNN_PARAMS = {
    "n_neighbors": 5,
    "weights": "distance",
    "metric": "minkowski",
    "n_jobs": -1
}

# Logistic Regression
LOGREG_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": RANDOM_SEED
}

# Gradient Boosting
GB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": RANDOM_SEED
}

# =============================================================================
# KONFIGURÁCIA DEEP LEARNING MODELOV
# =============================================================================

# Všeobecné parametre pre neurónové siete
DL_EPOCHS = 100
DL_BATCH_SIZE = 32
DL_LEARNING_RATE = 0.001
DL_EARLY_STOPPING_PATIENCE = 20

# LSTM (Long Short-Term Memory)
LSTM_PARAMS = {
    "units": [50, 50],           # Počet jednotiek v každej LSTM vrstve
    "dropout": 0.2,              # Dropout rate
    "sequence_length": 60,       # Dĺžka vstupnej sekvencie (60 dní)
}

# GRU (Gated Recurrent Unit)
GRU_PARAMS = {
    "units": [50, 50],
    "dropout": 0.2,
    "sequence_length": 60,
}

# =============================================================================
# KONFIGURÁCIA SIMULÁCIE
# =============================================================================

# Monte Carlo simulácia
MONTE_CARLO_SIMULATIONS = 100000  # Počet simulácií
MONTE_CARLO_DAYS = 252            # Počet dní na simuláciu (1 obchodný rok)

# Hybridný prístup - fitovanie rozdelenia výnosov
# Kandidátske distribúcie (scipy.stats názvy) testované pri identifikácii rozdelenia
# Výber prebieha podľa AIC (Akaike Information Criterion) + KS testu
DISTRIBUTION_CANDIDATES = [
    'norm',      # Normálne (Gaussovo) rozdelenie - základ GBM
    't',         # Studentovo t-rozdelenie - tučné chvosty
    'skewnorm',  # Skreslené normálne rozdelenie - asymetria
    'laplace',   # Laplaceovo rozdelenie - ostré maximum, tučné chvosty
    'johnsonsu', # Johnson SU rozdelenie - veľmi flexibilné
    'logistic',  # Logistické rozdelenie
    'gennorm',   # Zovšeobecnené normálne rozdelenie
    'nct',       # Necentralizované t-rozdelenie
]

# Metóda odhadu driftu (μ) pre Monte Carlo simulácie
# 'risk_neutral'  – drift = bezriziková miera (štandardný finančný prístup)
# 'historical'    – drift = priemer historických log-výnosov (citlivé na obdobie)
# 'custom'        – drift odvodený z SIMULATION_EXPECTED_ANNUAL_RETURN
SIMULATION_DRIFT_METHOD = 'historical'

# Očakávaný ročný výnos (desatinne, napr. 0.10 = 10 %) – používa sa len
# ak SIMULATION_DRIFT_METHOD == 'custom'
SIMULATION_EXPECTED_ANNUAL_RETURN = 0.10

# Kontrola extrémov v simuláciách
# Maximálny povolený denný log-výnos (abs. hodnota); väčšie budú orezané
SIMULATION_MAX_DAILY_LOG_RETURN = 0.2   # ≈ ±28 % denná zmena
# Minimálna povolená cena (ochrana pred numerickým kolapsom)
SIMULATION_MIN_PRICE = 0.01
# Prahové hodnoty pre sanity check reportu
SANITY_MAX_ANNUAL_RETURN_PCT = 100.0     # Priemerný ročný výnos > 100 % → varovanie
SANITY_MIN_ANNUAL_RETURN_PCT = -80.0     # Priemerný ročný výnos < -80 % → varovanie
SANITY_MAX_PRICE_RATIO = 50.0            # Konečná cena / počiatočná > 50× → varovanie

# Backtesting
INITIAL_CAPITAL = 200000        # Počiatočný kapitál v USD
TRANSACTION_COST = 0.001        # Transakčné náklady (0.1%)
RISK_FREE_RATE = 0.02           # Bezriziková úroková miera (2% ročne)

# =============================================================================
# KONFIGURÁCIA VIZUALIZÁCIÍ
# =============================================================================

# Štýl grafov
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_DPI = 100
FIGURE_SIZE = (14, 8)

# Farby pre grafy
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "neutral": "#7f7f7f",
    "train": "#1f77b4",
    "validation": "#ff7f0e",
    "test": "#2ca02c"
}

# =============================================================================
# KONFIGURÁCIA LOGOVANIA
# =============================================================================

# Úroveň logovania
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Formát logovania
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# =============================================================================
# POMOCNÉ FUNKCIE
# =============================================================================

def get_ticker_list():
    """
    Vráti zoznam tickerov na analýzu.
    
    Returns:
        list: Zoznam akciových tickerov
    """
    return STOCK_TICKERS.copy()


def get_date_range():
    """
    Vráti časové obdobie pre analýzu.
    
    Returns:
        tuple: (počiatočný_dátum, koncový_dátum)
    """
    return START_DATE, END_DATE


def print_config():
    """
    Vypíše aktuálnu konfiguráciu projektu.
    """
    print("=" * 60)
    print("KONFIGURÁCIA PROJEKTU")
    print("=" * 60)
    print(f"\nTickery na analýzu: {', '.join(STOCK_TICKERS)}")
    print(f"Časové obdobie: {START_DATE} až {END_DATE}")
    print(f"Predvolený ticker: {DEFAULT_TICKER}")
    print(f"\nRozdelenie dát:")
    print(f"  - Trénovacia množina: {TRAIN_RATIO*100:.0f}%")
    print(f"  - Validačná množina: {VAL_RATIO*100:.0f}%")
    print(f"  - Testovacia množina: {TEST_RATIO*100:.0f}%")
    print(f"\nRandom seed: {RANDOM_SEED}")
    print("=" * 60)


# Ak je modul spustený priamo, vypíše konfiguráciu
if __name__ == "__main__":
    print_config()
