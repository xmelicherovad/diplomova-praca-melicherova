# -*- coding: utf-8 -*-
"""
Modul pre tvorbu features z finančných dát.

Tento modul poskytuje funkcie na výpočet technických indikátorov:
- Kĺzavé priemery (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- Stochastic Oscillator
- Volume-based indikátory
- Lagged features (oneskorené hodnoty)

Autor: Dominika Melicherová
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import os

# Import konfigurácie
import config


# =============================================================================
# KĹZAVÉ PRIEMERY
# =============================================================================

def calculate_sma(
    df: pd.DataFrame,
    column: str = "Close",
    periods: List[int] = None
) -> pd.DataFrame:
    """
    Vypočíta jednoduchý kĺzavý priemer (Simple Moving Average).
    
    SMA je aritmetický priemer cien za určité obdobie.
    Používa sa na vyhladzovanie cenových dát a identifikáciu trendov.
    
    Args:
        df (pd.DataFrame): DataFrame s cenovými dátami
        column (str): Stĺpec pre výpočet
        periods (List[int], optional): Zoznam periód pre SMA
        
    Returns:
        pd.DataFrame: DataFrame s pridanými SMA stĺpcami
    """
    if periods is None:
        periods = config.SMA_PERIODS
    
    df_sma = df.copy()
    
    for period in periods:
        col_name = f"SMA_{period}"
        df_sma[col_name] = df_sma[column].rolling(window=period).mean()
    
    return df_sma


def calculate_ema(
    df: pd.DataFrame,
    column: str = "Close",
    periods: List[int] = None
) -> pd.DataFrame:
    """
    Vypočíta exponenciálny kĺzavý priemer (Exponential Moving Average).
    
    EMA dáva väčšiu váhu novším dátam, čím rýchlejšie reaguje na zmeny cien.
    
    Args:
        df (pd.DataFrame): DataFrame s cenovými dátami
        column (str): Stĺpec pre výpočet
        periods (List[int], optional): Zoznam periód pre EMA
        
    Returns:
        pd.DataFrame: DataFrame s pridanými EMA stĺpcami
    """
    if periods is None:
        periods = config.EMA_PERIODS
    
    df_ema = df.copy()
    
    for period in periods:
        col_name = f"EMA_{period}"
        df_ema[col_name] = df_ema[column].ewm(span=period, adjust=False).mean()
    
    return df_ema


# =============================================================================
# RSI (RELATIVE STRENGTH INDEX)
# =============================================================================

def calculate_rsi(
    df: pd.DataFrame,
    column: str = "Close",
    period: int = None
) -> pd.DataFrame:
    """
    Vypočíta RSI (Relative Strength Index).
    
    RSI meria rýchlosť a zmenu cenových pohybov.
    Hodnoty nad 70 indikujú prekúpenosť, pod 30 prepredanosť.
    
    Vzorec:
        RSI = 100 - (100 / (1 + RS))
        RS = Priemerný zisk / Priemerná strata
    
    Args:
        df (pd.DataFrame): DataFrame s cenovými dátami
        column (str): Stĺpec pre výpočet
        period (int, optional): Perióda pre výpočet RSI
        
    Returns:
        pd.DataFrame: DataFrame s pridaným RSI stĺpcom
    """
    if period is None:
        period = config.RSI_PERIOD
    
    df_rsi = df.copy()
    
    # Výpočet denných zmien
    delta = df_rsi[column].diff()
    
    # Oddelenie ziskov a strát
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    # Výpočet priemerného zisku a straty (EMA)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    # Výpočet RS a RSI
    rs = avg_gain / avg_loss
    df_rsi['RSI'] = 100 - (100 / (1 + rs))
    
    # Pridanie signálov prekúpenosti/prepredanosti
    df_rsi['RSI_Overbought'] = (df_rsi['RSI'] > config.RSI_OVERBOUGHT).astype(int)
    df_rsi['RSI_Oversold'] = (df_rsi['RSI'] < config.RSI_OVERSOLD).astype(int)
    
    return df_rsi


# =============================================================================
# MACD (MOVING AVERAGE CONVERGENCE DIVERGENCE)
# =============================================================================

def calculate_macd(
    df: pd.DataFrame,
    column: str = "Close",
    fast_period: int = None,
    slow_period: int = None,
    signal_period: int = None
) -> pd.DataFrame:
    """
    Vypočíta MACD (Moving Average Convergence Divergence).
    
    MACD je trend-nasledujúci momentový indikátor.
    - MACD Line = EMA(fast) - EMA(slow)
    - Signal Line = EMA(MACD Line)
    - Histogram = MACD Line - Signal Line
    
    Args:
        df (pd.DataFrame): DataFrame s cenovými dátami
        column (str): Stĺpec pre výpočet
        fast_period (int, optional): Rýchla EMA perióda
        slow_period (int, optional): Pomalá EMA perióda
        signal_period (int, optional): Signálna perióda
        
    Returns:
        pd.DataFrame: DataFrame s pridanými MACD stĺpcami
    """
    if fast_period is None:
        fast_period = config.MACD_FAST_PERIOD
    if slow_period is None:
        slow_period = config.MACD_SLOW_PERIOD
    if signal_period is None:
        signal_period = config.MACD_SIGNAL_PERIOD
    
    df_macd = df.copy()
    
    # Výpočet EMA
    ema_fast = df_macd[column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df_macd[column].ewm(span=slow_period, adjust=False).mean()
    
    # MACD linka
    df_macd['MACD'] = ema_fast - ema_slow
    
    # Signálna linka
    df_macd['MACD_Signal'] = df_macd['MACD'].ewm(span=signal_period, adjust=False).mean()
    
    # Histogram
    df_macd['MACD_Histogram'] = df_macd['MACD'] - df_macd['MACD_Signal']
    
    # Signály kríženia
    df_macd['MACD_CrossOver'] = (
        (df_macd['MACD'] > df_macd['MACD_Signal']) & 
        (df_macd['MACD'].shift(1) <= df_macd['MACD_Signal'].shift(1))
    ).astype(int)
    
    df_macd['MACD_CrossUnder'] = (
        (df_macd['MACD'] < df_macd['MACD_Signal']) & 
        (df_macd['MACD'].shift(1) >= df_macd['MACD_Signal'].shift(1))
    ).astype(int)
    
    return df_macd


# =============================================================================
# BOLLINGER BANDS
# =============================================================================

def calculate_bollinger_bands(
    df: pd.DataFrame,
    column: str = "Close",
    period: int = None,
    std_dev: float = None
) -> pd.DataFrame:
    """
    Vypočíta Bollinger Bands.
    
    Bollinger Bands pozostávajú z:
    - Stredné pásmo (SMA)
    - Horné pásmo (SMA + n*štandardná odchýlka)
    - Dolné pásmo (SMA - n*štandardná odchýlka)
    
    Používajú sa na meranie volatility a identifikáciu prekúpených/prepredaných stavov.
    
    Args:
        df (pd.DataFrame): DataFrame s cenovými dátami
        column (str): Stĺpec pre výpočet
        period (int, optional): Perióda pre SMA
        std_dev (float, optional): Počet štandardných odchýlok
        
    Returns:
        pd.DataFrame: DataFrame s pridanými Bollinger Bands stĺpcami
    """
    if period is None:
        period = config.BOLLINGER_PERIOD
    if std_dev is None:
        std_dev = config.BOLLINGER_STD
    
    df_bb = df.copy()
    
    # Stredné pásmo (SMA)
    df_bb['BB_Middle'] = df_bb[column].rolling(window=period).mean()
    
    # Štandardná odchýlka
    rolling_std = df_bb[column].rolling(window=period).std()
    
    # Horné a dolné pásmo
    df_bb['BB_Upper'] = df_bb['BB_Middle'] + (std_dev * rolling_std)
    df_bb['BB_Lower'] = df_bb['BB_Middle'] - (std_dev * rolling_std)
    
    # Šírka pásma (Bandwidth)
    df_bb['BB_Bandwidth'] = (df_bb['BB_Upper'] - df_bb['BB_Lower']) / df_bb['BB_Middle']
    
    # Pozícia ceny v rámci pásiem (%B)
    df_bb['BB_PercentB'] = (
        (df_bb[column] - df_bb['BB_Lower']) / 
        (df_bb['BB_Upper'] - df_bb['BB_Lower'])
    )
    
    return df_bb


# =============================================================================
# ATR (AVERAGE TRUE RANGE)
# =============================================================================

def calculate_atr(
    df: pd.DataFrame,
    period: int = None
) -> pd.DataFrame:
    """
    Vypočíta ATR (Average True Range).
    
    ATR meria volatilitu trhu. True Range je maximum z:
    - High - Low
    - |High - Previous Close|
    - |Low - Previous Close|
    
    Args:
        df (pd.DataFrame): DataFrame s OHLC dátami
        period (int, optional): Perióda pre ATR
        
    Returns:
        pd.DataFrame: DataFrame s pridaným ATR stĺpcom
    """
    if period is None:
        period = config.ATR_PERIOD
    
    df_atr = df.copy()
    
    # Výpočet True Range
    high_low = df_atr['High'] - df_atr['Low']
    high_close = abs(df_atr['High'] - df_atr['Close'].shift(1))
    low_close = abs(df_atr['Low'] - df_atr['Close'].shift(1))
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_atr['True_Range'] = tr
    
    # ATR (EMA of True Range)
    df_atr['ATR'] = tr.ewm(span=period, adjust=False).mean()
    
    # ATR ako percento ceny
    df_atr['ATR_Percent'] = (df_atr['ATR'] / df_atr['Close']) * 100
    
    return df_atr


# =============================================================================
# STOCHASTIC OSCILLATOR
# =============================================================================

def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int = None,
    d_period: int = None
) -> pd.DataFrame:
    """
    Vypočíta Stochastic Oscillator.
    
    Stochastic porovnáva uzatváraciu cenu s cenovým rozsahom za dané obdobie.
    - %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    - %D = SMA(%K)
    
    Hodnoty nad 80 = prekúpenosť, pod 20 = prepredanosť
    
    Args:
        df (pd.DataFrame): DataFrame s OHLC dátami
        k_period (int, optional): Perióda pre %K
        d_period (int, optional): Perióda pre %D (vyhladenie)
        
    Returns:
        pd.DataFrame: DataFrame s pridanými Stochastic stĺpcami
    """
    if k_period is None:
        k_period = config.STOCH_K_PERIOD
    if d_period is None:
        d_period = config.STOCH_D_PERIOD
    
    df_stoch = df.copy()
    
    # Najnižšia a najvyššia cena za periódu
    lowest_low = df_stoch['Low'].rolling(window=k_period).min()
    highest_high = df_stoch['High'].rolling(window=k_period).max()
    
    # %K
    df_stoch['Stoch_K'] = (
        (df_stoch['Close'] - lowest_low) / 
        (highest_high - lowest_low) * 100
    )
    
    # %D (vyhladený %K)
    df_stoch['Stoch_D'] = df_stoch['Stoch_K'].rolling(window=d_period).mean()
    
    # Signály
    df_stoch['Stoch_Overbought'] = (df_stoch['Stoch_K'] > 80).astype(int)
    df_stoch['Stoch_Oversold'] = (df_stoch['Stoch_K'] < 20).astype(int)
    
    return df_stoch


# =============================================================================
# VOLUME-BASED INDIKÁTORY
# =============================================================================

def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vypočíta indikátory založené na objeme obchodov.
    
    Obsahuje:
    - OBV (On-Balance Volume)
    - Volume SMA
    - Volume Ratio (relatívny objem)
    - Price-Volume Trend
    
    Args:
        df (pd.DataFrame): DataFrame s OHLCV dátami
        
    Returns:
        pd.DataFrame: DataFrame s pridanými volume indikátormi
    """
    df_vol = df.copy()
    
    # On-Balance Volume (OBV)
    # OBV kumuluje objem podľa smeru ceny
    price_change = df_vol['Close'].diff()
    obv = []
    obv_value = 0
    
    for i, change in enumerate(price_change):
        if pd.isna(change):
            obv.append(0)
        elif change > 0:
            obv_value += df_vol['Volume'].iloc[i]
            obv.append(obv_value)
        elif change < 0:
            obv_value -= df_vol['Volume'].iloc[i]
            obv.append(obv_value)
        else:
            obv.append(obv_value)
    
    df_vol['OBV'] = obv
    
    # Volume SMA
    df_vol['Volume_SMA_20'] = df_vol['Volume'].rolling(window=20).mean()
    
    # Volume Ratio (aktuálny objem / priemerný objem)
    df_vol['Volume_Ratio'] = df_vol['Volume'] / df_vol['Volume_SMA_20']
    
    # Price-Volume Trend (PVT)
    # Podobné OBV, ale váži objem podľa percentuálnej zmeny ceny
    df_vol['PVT'] = (
        ((df_vol['Close'] - df_vol['Close'].shift(1)) / df_vol['Close'].shift(1)) * 
        df_vol['Volume']
    ).cumsum()
    
    # Accumulation/Distribution Line
    clv = ((df_vol['Close'] - df_vol['Low']) - (df_vol['High'] - df_vol['Close'])) / \
          (df_vol['High'] - df_vol['Low'])
    clv = clv.fillna(0)
    df_vol['AD_Line'] = (clv * df_vol['Volume']).cumsum()
    
    return df_vol


# =============================================================================
# LAGGED FEATURES
# =============================================================================

def create_lagged_features(
    df: pd.DataFrame,
    columns: List[str] = None,
    lag_periods: List[int] = None
) -> pd.DataFrame:
    """
    Vytvorí oneskorené (lagged) features.
    
    Lagged features sú hodnoty z predchádzajúcich období,
    ktoré môžu byť užitočné pre predikciu.
    
    Args:
        df (pd.DataFrame): DataFrame s dátami
        columns (List[str], optional): Stĺpce pre lag features
        lag_periods (List[int], optional): Periódy oneskorenia
        
    Returns:
        pd.DataFrame: DataFrame s pridanými lag features
    """
    if columns is None:
        columns = ['Close', 'Volume', 'Return_1d']
    
    if lag_periods is None:
        lag_periods = config.LAG_PERIODS
    
    df_lag = df.copy()
    
    for col in columns:
        if col not in df_lag.columns:
            continue
            
        for lag in lag_periods:
            lag_col_name = f"{col}_Lag_{lag}"
            df_lag[lag_col_name] = df_lag[col].shift(lag)
    
    return df_lag


# =============================================================================
# DODATOČNÉ FEATURES
# =============================================================================

def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vypočíta dodatočné cenové features.
    
    Args:
        df (pd.DataFrame): DataFrame s OHLC dátami
        
    Returns:
        pd.DataFrame: DataFrame s pridanými cenovými features
    """
    df_price = df.copy()
    
    # Cenový rozsah (High - Low)
    df_price['Price_Range'] = df_price['High'] - df_price['Low']
    
    # Cenový rozsah ako percento
    df_price['Price_Range_Pct'] = (df_price['Price_Range'] / df_price['Close']) * 100
    
    # Gap (Open - Previous Close)
    df_price['Gap'] = df_price['Open'] - df_price['Close'].shift(1)
    df_price['Gap_Pct'] = (df_price['Gap'] / df_price['Close'].shift(1)) * 100
    
    # Intraday return (Close - Open)
    df_price['Intraday_Return'] = (
        (df_price['Close'] - df_price['Open']) / df_price['Open'] * 100
    )
    
    # Pozícia close v dennom rozsahu
    df_price['Close_Position'] = (
        (df_price['Close'] - df_price['Low']) / 
        (df_price['High'] - df_price['Low'])
    )
    
    # Vzdialenosť od kĺzavých priemerov
    if 'SMA_20' in df_price.columns:
        df_price['Distance_SMA_20'] = (
            (df_price['Close'] - df_price['SMA_20']) / df_price['SMA_20'] * 100
        )
    
    if 'SMA_50' in df_price.columns:
        df_price['Distance_SMA_50'] = (
            (df_price['Close'] - df_price['SMA_50']) / df_price['SMA_50'] * 100
        )
    
    return df_price


def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vypočíta momentové indikátory.
    
    Args:
        df (pd.DataFrame): DataFrame s cenovými dátami
        
    Returns:
        pd.DataFrame: DataFrame s pridanými momentovými indikátormi
    """
    df_mom = df.copy()
    
    # Rate of Change (ROC)
    for period in [5, 10, 20]:
        df_mom[f'ROC_{period}'] = (
            (df_mom['Close'] - df_mom['Close'].shift(period)) / 
            df_mom['Close'].shift(period) * 100
        )
    
    # Momentum
    for period in [10, 20]:
        df_mom[f'Momentum_{period}'] = df_mom['Close'] - df_mom['Close'].shift(period)
    
    # Williams %R
    period = 14
    highest_high = df_mom['High'].rolling(window=period).max()
    lowest_low = df_mom['Low'].rolling(window=period).min()
    df_mom['Williams_R'] = (
        (highest_high - df_mom['Close']) / (highest_high - lowest_low) * -100
    )
    
    # CCI (Commodity Channel Index)
    period = 20
    typical_price = (df_mom['High'] + df_mom['Low'] + df_mom['Close']) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x)))
    )
    df_mom['CCI'] = (typical_price - sma_tp) / (0.015 * mean_deviation)
    
    return df_mom


def add_time_features(df: pd.DataFrame, date_column: str = 'Datum') -> pd.DataFrame:
    """
    Pridá časové features.
    
    Args:
        df (pd.DataFrame): DataFrame s dátumovým stĺpcom
        date_column (str): Názov dátumového stĺpca
        
    Returns:
        pd.DataFrame: DataFrame s pridanými časovými features
    """
    df_time = df.copy()
    
    # Konverzia na datetime ak je potrebné
    if date_column in df_time.columns:
        df_time[date_column] = pd.to_datetime(df_time[date_column], utc=True).dt.tz_localize(None)
        
        # Deň v týždni (0 = pondelok, 4 = piatok)
        df_time['DayOfWeek'] = df_time[date_column].dt.dayofweek
        
        # Deň v mesiaci
        df_time['DayOfMonth'] = df_time[date_column].dt.day
        
        # Mesiac
        df_time['Month'] = df_time[date_column].dt.month
        
        # Štvrťrok
        df_time['Quarter'] = df_time[date_column].dt.quarter
        
        # Rok
        df_time['Year'] = df_time[date_column].dt.year
        
        # Je to koniec mesiaca?
        df_time['IsMonthEnd'] = df_time[date_column].dt.is_month_end.astype(int)
        
        # Je to začiatok mesiaca?
        df_time['IsMonthStart'] = df_time[date_column].dt.is_month_start.astype(int)
    
    return df_time


# =============================================================================
# HLAVNÝ PIPELINE PRE FEATURE ENGINEERING
# =============================================================================

def create_all_features(
    df: pd.DataFrame,
    ticker: str = None,
    save_to_file: bool = True
) -> pd.DataFrame:
    """
    Vytvorí všetky features pre dané dáta.
    
    Táto funkcia kombinuje všetky feature engineering operácie
    do jedného kompletného pipeline.
    
    Args:
        df (pd.DataFrame): Predspracované dáta
        ticker (str, optional): Symbol akcie
        save_to_file (bool): Ak True, uloží dáta s features
        
    Returns:
        pd.DataFrame: DataFrame so všetkými features
    """
    print("\n" + "=" * 60)
    print(f"FEATURE ENGINEERING" + (f" PRE {ticker}" if ticker else ""))
    print("=" * 60)
    
    df_features = df.copy()
    
    # 1. Kĺzavé priemery
    print("[INFO] Výpočet kĺzavých priemerov...")
    df_features = calculate_sma(df_features)
    df_features = calculate_ema(df_features)
    
    # 2. RSI
    print("[INFO] Výpočet RSI...")
    df_features = calculate_rsi(df_features)
    
    # 3. MACD
    print("[INFO] Výpočet MACD...")
    df_features = calculate_macd(df_features)
    
    # 4. Bollinger Bands
    print("[INFO] Výpočet Bollinger Bands...")
    df_features = calculate_bollinger_bands(df_features)
    
    # 5. ATR
    print("[INFO] Výpočet ATR...")
    df_features = calculate_atr(df_features)
    
    # 6. Stochastic
    print("[INFO] Výpočet Stochastic Oscillator...")
    df_features = calculate_stochastic(df_features)
    
    # 7. Volume indikátory
    print("[INFO] Výpočet volume indikátorov...")
    df_features = calculate_volume_indicators(df_features)
    
    # 8. Cenové features
    print("[INFO] Výpočet cenových features...")
    df_features = calculate_price_features(df_features)
    
    # 9. Momentové indikátory
    print("[INFO] Výpočet momentových indikátorov...")
    df_features = calculate_momentum_indicators(df_features)
    
    # 10. Lagged features
    print("[INFO] Vytváranie lagged features...")
    columns_for_lag = ['Close', 'Volume', 'RSI', 'MACD']
    existing_cols = [c for c in columns_for_lag if c in df_features.columns]
    df_features = create_lagged_features(df_features, columns=existing_cols)
    
    # 11. Časové features
    print("[INFO] Pridávanie časových features...")
    df_features = add_time_features(df_features)
    
    # Odstránenie riadkov s NaN
    initial_len = len(df_features)
    df_features = df_features.dropna()
    print(f"[INFO] Odstránených {initial_len - len(df_features)} riadkov s NaN")
    
    print(f"\n[OK] Vytvorených {len(df_features.columns)} stĺpcov (features)")
    
    # Uloženie
    if save_to_file and ticker:
        filepath = os.path.join(config.DATA_DIR, f"{ticker}_features.csv")
        df_features.to_csv(filepath, index=False)
        print(f"[OK] Features uložené do {filepath}")
    
    print("=" * 60)
    
    return df_features


def get_feature_list(df: pd.DataFrame, exclude_targets: bool = True) -> List[str]:
    """
    Vráti zoznam feature stĺpcov (bez cieľových premenných).
    
    Args:
        df (pd.DataFrame): DataFrame s features
        exclude_targets (bool): Ak True, vylúči Target_ stĺpce
        
    Returns:
        List[str]: Zoznam názvov feature stĺpcov
    """
    # Stĺpce na vylúčenie
    exclude_cols = ['Datum', 'Ticker', 'Date']
    
    if exclude_targets:
        exclude_cols.extend([col for col in df.columns if col.startswith('Target_')])
    
    # Vybrať len numerické stĺpce
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filtrovanie
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    return feature_cols


def get_feature_importance_summary(df: pd.DataFrame) -> Dict:
    """
    Vytvorí súhrn dostupných features podľa kategórií.
    
    Args:
        df (pd.DataFrame): DataFrame s features
        
    Returns:
        Dict: Slovník s kategóriami features
    """
    columns = df.columns.tolist()
    
    summary = {
        'klzave_priemery': [c for c in columns if c.startswith(('SMA_', 'EMA_'))],
        'rsi': [c for c in columns if 'RSI' in c],
        'macd': [c for c in columns if 'MACD' in c],
        'bollinger': [c for c in columns if 'BB_' in c],
        'atr': [c for c in columns if 'ATR' in c],
        'stochastic': [c for c in columns if 'Stoch' in c],
        'volume': [c for c in columns if c in ['OBV', 'Volume_SMA_20', 'Volume_Ratio', 'PVT', 'AD_Line']],
        'momentum': [c for c in columns if c.startswith(('ROC_', 'Momentum_', 'Williams', 'CCI'))],
        'lagged': [c for c in columns if '_Lag_' in c],
        'casove': [c for c in columns if c in ['DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'Year', 'IsMonthEnd', 'IsMonthStart']],
        'cenove': [c for c in columns if c in ['Price_Range', 'Price_Range_Pct', 'Gap', 'Gap_Pct', 'Intraday_Return', 'Close_Position']],
    }
    
    return summary


# =============================================================================
# HLAVNÁ FUNKCIA PRE TESTOVANIE MODULU
# =============================================================================

if __name__ == "__main__":
    import data_downloader
    import data_preprocessing
    
    print("=" * 60)
    print("TESTOVANIE MODULU FEATURE_ENGINEERING")
    print("=" * 60)
    
    # Načítanie a predspracovanie dát
    try:
        df = data_downloader.load_stock_data("AAPL")
    except FileNotFoundError:
        print("[INFO] Dáta nie sú k dispozícii, sťahujem...")
        df = data_downloader.download_stock_data("AAPL")
    
    # Predspracovanie
    df_processed = data_preprocessing.preprocess_pipeline(df, ticker="AAPL", save_to_file=False)
    
    # Feature engineering
    df_features = create_all_features(df_processed, ticker="AAPL")
    
    # Súhrn features
    print("\nSúhrn features podľa kategórií:")
    summary = get_feature_importance_summary(df_features)
    for category, features in summary.items():
        print(f"  {category}: {len(features)} features")
    
    # Celkový počet features
    feature_list = get_feature_list(df_features)
    print(f"\nCelkový počet features: {len(feature_list)}")
