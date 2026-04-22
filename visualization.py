# -*- coding: utf-8 -*-
"""
Modul pre vizualizáciu finančných dát a výsledkov modelov.

Tento modul poskytuje funkcie na vytvorenie rôznych grafov:
- Cenové grafy (OHLC, sviečkové grafy)
- Technické indikátory
- Distribúcie výnosov
- Korelačné matice
- Porovnanie modelov
- Backtesting výsledky

Autor: Dominika Melicherová
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List, Dict, Optional, Tuple

# Import konfigurácie
import config

# Nastavenie štýlu grafov
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = config.FIGURE_SIZE
plt.rcParams['figure.dpi'] = config.FIGURE_DPI
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


# =============================================================================
# ZÁKLADNÉ CENOVÉ GRAFY
# =============================================================================

def plot_price_history(
    df: pd.DataFrame,
    ticker: str = None,
    date_column: str = 'Datum',
    price_column: str = 'Close',
    title: str = None,
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí historický vývoj ceny akcie.
    
    Args:
        df (pd.DataFrame): DataFrame s dátami
        ticker (str, optional): Symbol akcie pre titulok
        date_column (str): Názov dátumového stĺpca
        price_column (str): Názov cenového stĺpca
        title (str, optional): Vlastný titulok
        save_path (str, optional): Cesta pre uloženie grafu
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    
    # Konverzia dátumu
    dates = pd.to_datetime(df[date_column])
    
    # Vykreslenie ceny
    ax.plot(dates, df[price_column], color=config.COLORS['primary'], linewidth=1.5)
    
    # Formátovanie
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Historický vývoj ceny{" - " + ticker if ticker else ""}')
    
    ax.set_xlabel('Dátum')
    ax.set_ylabel(f'Cena ({price_column})')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    # Mriežka
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Uloženie
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


def plot_ohlc(
    df: pd.DataFrame,
    ticker: str = None,
    date_column: str = 'Datum',
    last_n_days: int = 60,
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí OHLC (Open-High-Low-Close) graf.
    
    Args:
        df (pd.DataFrame): DataFrame s OHLC dátami
        ticker (str, optional): Symbol akcie
        date_column (str): Názov dátumového stĺpca
        last_n_days (int): Počet posledných dní na zobrazenie
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    # Použitie posledných N dní
    df_plot = df.tail(last_n_days).copy()
    df_plot[date_column] = pd.to_datetime(df_plot[date_column])
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    
    # Vykreslenie OHLC
    for idx, row in df_plot.iterrows():
        color = config.COLORS['positive'] if row['Close'] >= row['Open'] else config.COLORS['negative']
        
        # Vertikálna čiara (High-Low)
        ax.plot([row[date_column], row[date_column]], 
                [row['Low'], row['High']], 
                color=color, linewidth=1)
        
        # Horizontálne čiarky (Open a Close)
        ax.plot([row[date_column], row[date_column]], 
                [row['Open'], row['Close']], 
                color=color, linewidth=4)
    
    ax.set_title(f'OHLC graf{" - " + ticker if ticker else ""} (posledných {last_n_days} dní)')
    ax.set_xlabel('Dátum')
    ax.set_ylabel('Cena')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


def plot_candlestick(
    df: pd.DataFrame,
    ticker: str = None,
    date_column: str = 'Datum',
    last_n_days: int = 60,
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí sviečkový graf (candlestick chart).
    
    Args:
        df (pd.DataFrame): DataFrame s OHLC dátami
        ticker (str, optional): Symbol akcie
        date_column (str): Názov dátumového stĺpca
        last_n_days (int): Počet posledných dní
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    df_plot = df.tail(last_n_days).copy().reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    
    width = 0.6
    
    for i, row in df_plot.iterrows():
        if row['Close'] >= row['Open']:
            color = config.COLORS['positive']
            bottom = row['Open']
            height = row['Close'] - row['Open']
        else:
            color = config.COLORS['negative']
            bottom = row['Close']
            height = row['Open'] - row['Close']
        
        # Telo sviečky
        ax.add_patch(Rectangle((i - width/2, bottom), width, height, 
                                facecolor=color, edgecolor=color))
        
        # Knôty
        ax.plot([i, i], [row['Low'], min(row['Open'], row['Close'])], 
                color=color, linewidth=1)
        ax.plot([i, i], [max(row['Open'], row['Close']), row['High']], 
                color=color, linewidth=1)
    
    # X-osi labels
    step = max(1, len(df_plot) // 10)
    ax.set_xticks(range(0, len(df_plot), step))
    
    if date_column in df_plot.columns:
        dates = pd.to_datetime(df_plot[date_column])
        ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates[::step]], rotation=45)
    
    ax.set_title(f'Sviečkový graf{" - " + ticker if ticker else ""} (posledných {last_n_days} dní)')
    ax.set_xlabel('Dátum')
    ax.set_ylabel('Cena')
    ax.grid(True, alpha=0.3)
    ax.autoscale_view()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


# =============================================================================
# GRAFY S TECHNICKÝMI INDIKÁTORMI
# =============================================================================

def plot_price_with_ma(
    df: pd.DataFrame,
    ticker: str = None,
    date_column: str = 'Datum',
    ma_columns: List[str] = None,
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí cenu s kĺzavými priemermi.
    
    Args:
        df (pd.DataFrame): DataFrame s dátami
        ticker (str, optional): Symbol akcie
        date_column (str): Názov dátumového stĺpca
        ma_columns (List[str], optional): Zoznam MA stĺpcov
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    if ma_columns is None:
        ma_columns = [col for col in df.columns if col.startswith(('SMA_', 'EMA_'))]
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    
    dates = pd.to_datetime(df[date_column])
    
    # Cena
    ax.plot(dates, df['Close'], label='Uzatváracia cena', 
            color=config.COLORS['primary'], linewidth=1.5)
    
    # Kĺzavé priemery
    colors = plt.cm.tab10(np.linspace(0, 1, len(ma_columns)))
    for col, color in zip(ma_columns, colors):
        if col in df.columns:
            ax.plot(dates, df[col], label=col, linewidth=1, alpha=0.7)
    
    ax.set_title(f'Cena s kĺzavými priemermi{" - " + ticker if ticker else ""}')
    ax.set_xlabel('Dátum')
    ax.set_ylabel('Cena')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


def plot_bollinger_bands(
    df: pd.DataFrame,
    ticker: str = None,
    date_column: str = 'Datum',
    last_n_days: int = 252,
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí cenu s Bollinger Bands.
    
    Args:
        df (pd.DataFrame): DataFrame s BB dátami
        ticker (str, optional): Symbol akcie
        date_column (str): Názov dátumového stĺpca
        last_n_days (int): Počet dní na zobrazenie
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    df_plot = df.tail(last_n_days).copy()
    dates = pd.to_datetime(df_plot[date_column])
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    
    # Bollinger Bands
    ax.fill_between(dates, df_plot['BB_Lower'], df_plot['BB_Upper'], 
                    alpha=0.2, color=config.COLORS['primary'], label='Bollinger pásmo')
    ax.plot(dates, df_plot['BB_Middle'], '--', 
            color=config.COLORS['secondary'], label='BB stred (SMA)')
    ax.plot(dates, df_plot['BB_Upper'], 
            color=config.COLORS['primary'], alpha=0.5, linewidth=0.5)
    ax.plot(dates, df_plot['BB_Lower'], 
            color=config.COLORS['primary'], alpha=0.5, linewidth=0.5)
    
    # Cena
    ax.plot(dates, df_plot['Close'], 
            color=config.COLORS['primary'], linewidth=1.5, label='Uzatváracia cena')
    
    ax.set_title(f'Bollinger Bands{" - " + ticker if ticker else ""}')
    ax.set_xlabel('Dátum')
    ax.set_ylabel('Cena')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


def plot_technical_indicators(
    df: pd.DataFrame,
    ticker: str = None,
    date_column: str = 'Datum',
    last_n_days: int = 252,
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí komplexný graf s cenou a technickými indikátormi.
    
    Obsahuje:
    - Cena s kĺzavými priemermi
    - Volume
    - RSI
    - MACD
    
    Args:
        df (pd.DataFrame): DataFrame s technickými indikátormi
        ticker (str, optional): Symbol akcie
        date_column (str): Názov dátumového stĺpca
        last_n_days (int): Počet dní na zobrazenie
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    df_plot = df.tail(last_n_days).copy()
    dates = pd.to_datetime(df_plot[date_column])
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # 1. Cena s MA
    ax1 = axes[0]
    ax1.plot(dates, df_plot['Close'], label='Cena', color=config.COLORS['primary'])
    
    for col in ['SMA_20', 'SMA_50', 'EMA_12']:
        if col in df_plot.columns:
            ax1.plot(dates, df_plot[col], label=col, alpha=0.7, linewidth=1)
    
    ax1.set_title(f'Technická analýza{" - " + ticker if ticker else ""}')
    ax1.set_ylabel('Cena')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Volume
    ax2 = axes[1]
    colors = [config.COLORS['positive'] if df_plot['Close'].iloc[i] >= df_plot['Open'].iloc[i] 
              else config.COLORS['negative'] for i in range(len(df_plot))]
    ax2.bar(dates, df_plot['Volume'], color=colors, alpha=0.7)
    ax2.set_ylabel('Objem')
    ax2.grid(True, alpha=0.3)
    
    # 3. RSI
    ax3 = axes[2]
    if 'RSI' in df_plot.columns:
        ax3.plot(dates, df_plot['RSI'], color=config.COLORS['primary'])
        ax3.axhline(y=70, color=config.COLORS['negative'], linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color=config.COLORS['positive'], linestyle='--', alpha=0.5)
        ax3.fill_between(dates, 30, 70, alpha=0.1, color=config.COLORS['neutral'])
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    # 4. MACD
    ax4 = axes[3]
    if 'MACD' in df_plot.columns and 'MACD_Signal' in df_plot.columns:
        ax4.plot(dates, df_plot['MACD'], label='MACD', color=config.COLORS['primary'])
        ax4.plot(dates, df_plot['MACD_Signal'], label='Signálna línia', color=config.COLORS['secondary'])
        
        if 'MACD_Histogram' in df_plot.columns:
            colors_hist = [config.COLORS['positive'] if v >= 0 else config.COLORS['negative'] 
                          for v in df_plot['MACD_Histogram']]
            ax4.bar(dates, df_plot['MACD_Histogram'], color=colors_hist, alpha=0.5)
        
        ax4.axhline(y=0, color=config.COLORS['neutral'], linestyle='-', alpha=0.3)
        ax4.set_ylabel('MACD')
        ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Dátum')
    
    # Formátovanie x-osi
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


# =============================================================================
# DISTRIBÚCIA A ŠTATISTIKA
# =============================================================================

def plot_returns_distribution(
    df: pd.DataFrame,
    return_column: str = 'Return_1d',
    ticker: str = None,
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí distribúciu výnosov.
    
    Args:
        df (pd.DataFrame): DataFrame s výnosmi
        return_column (str): Názov stĺpca s výnosmi
        ticker (str, optional): Symbol akcie
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    returns = df[return_column].dropna()
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(returns, bins=50, color=config.COLORS['primary'], 
             alpha=0.7, edgecolor='black', density=True)
    
    # Normálna distribúcia pre porovnanie
    from scipy import stats
    x = np.linspace(returns.min(), returns.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), 
             'r-', linewidth=2, label='Normálna distribúcia')
    
    ax1.set_title(f'Distribúcia výnosov{" - " + ticker if ticker else ""}')
    ax1.set_xlabel('Výnos (%)')
    ax1.set_ylabel('Hustota')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # QQ plot
    ax2 = axes[1]
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q graf (porovnanie s normálnou distribúciou)')
    ax2.grid(True, alpha=0.3)
    
    # Štatistiky
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    fig.suptitle(f'Šikmosť: {skewness:.3f}, Špicatosť: {kurtosis:.3f}', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: List[str] = None,
    title: str = "Korelačná matica",
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí korelačnú maticu pre vybrané stĺpce.
    
    Args:
        df (pd.DataFrame): DataFrame s dátami
        columns (List[str], optional): Stĺpce pre koreláciu
        title (str): Titulok grafu
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    if columns is None:
        # Vyber numerické stĺpce, max 20
        columns = df.select_dtypes(include=[np.number]).columns.tolist()[:20]
    
    # Výpočet korelácie
    corr_matrix = df[columns].corr()
    
    # Veľkosť grafu podľa počtu stĺpcov
    size = max(10, len(columns) * 0.5)
    fig, ax = plt.subplots(figsize=(size, size))
    
    # Heatmapa
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=len(columns) <= 15, 
                cmap='RdYlBu_r', center=0, ax=ax,
                fmt='.2f', square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Dôležitosť features",
    top_n: int = 20,
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí dôležitosť features z modelu.
    
    Args:
        feature_names (List[str]): Názvy features
        importances (np.ndarray): Hodnoty dôležitosti
        title (str): Titulok
        top_n (int): Počet top features na zobrazenie
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    # Zoradenie podľa dôležitosti
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.3)))
    
    # Horizontálny bar chart
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importances[indices], color=config.COLORS['primary'], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    
    ax.set_xlabel('Dôležitosť')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


# =============================================================================
# VÝSLEDKY MODELOV
# =============================================================================

def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí porovnanie predikcií so skutočnými hodnotami.
    
    Args:
        y_true (np.ndarray): Skutočné hodnoty
        y_pred (np.ndarray): Predikované hodnoty
        model_name (str): Názov modelu
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, color=config.COLORS['primary'])
    
    # Ideálna línia
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideálna predikcia')
    
    ax1.set_xlabel('Skutočné hodnoty')
    ax1.set_ylabel('Predikované hodnoty')
    ax1.set_title(f'{model_name}: Predikcie vs. Skutočnosť')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Reziduá
    ax2 = axes[1]
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, color=config.COLORS['primary'])
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predikované hodnoty')
    ax2.set_ylabel('Reziduá')
    ax2.set_title('Reziduálny graf')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


def plot_time_series_prediction(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí časový rad predikcií a skutočných hodnôt.
    
    Args:
        dates (pd.Series): Dátumy
        y_true (np.ndarray): Skutočné hodnoty
        y_pred (np.ndarray): Predikované hodnoty
        model_name (str): Názov modelu
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    
    dates = pd.to_datetime(dates)
    
    ax.plot(dates, y_true, label='Skutočné hodnoty', 
            color=config.COLORS['primary'], linewidth=1.5)
    ax.plot(dates, y_pred, label='Predikcie', 
            color=config.COLORS['secondary'], linewidth=1.5, alpha=0.8)
    
    ax.set_title(f'{model_name}: Časový rad predikcií')
    ax.set_xlabel('Dátum')
    ax.set_ylabel('Hodnota')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


def plot_model_comparison(
    results: Dict[str, Dict],
    metric: str = 'accuracy',
    title: str = None,
    save_path: str = None,
    model_type: str = "classification"
) -> plt.Figure:
    """
    Vykreslí porovnanie výkonnosti modelov.
    
    Args:
        results (Dict[str, Dict]): Slovník s výsledkami modelov
        metric (str): Metrika na porovnanie
        title (str, optional): Titulok
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if model_type == "classification":
        colors = [config.COLORS['primary'] if v == max(values) else config.COLORS['secondary'] for v in values]
    elif model_type == "regression":
        colors = [config.COLORS['primary'] if v == min(values) else config.COLORS['secondary'] for v in values]
    
    bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Pridanie hodnôt nad stĺpce
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Porovnanie modelov - {metric}')
    
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.capitalize())
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    labels: List[str] = None,
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí konfúznu maticu.
    
    Args:
        y_true (np.ndarray): Skutočné triedy
        y_pred (np.ndarray): Predikované triedy
        model_name (str): Názov modelu
        labels (List[str], optional): Názvy tried
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = ['Pokles', 'Rast']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    
    ax.set_title(f'{model_name}: Konfúzna matica')
    ax.set_xlabel('Predikovaná trieda')
    ax.set_ylabel('Skutočná trieda')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


def plot_learning_curves(
    history: Dict,
    model_name: str = "Model",
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí krivky učenia (loss a metrika počas trénovania).
    
    Args:
        history (Dict): História trénovania (z Keras alebo podobné)
        model_name (str): Názov modelu
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1 = axes[0]
    if 'loss' in history:
        ax1.plot(history['loss'], label='Trénovacia strata', color=config.COLORS['train'])
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validačná strata', color=config.COLORS['validation'])
    ax1.set_title(f'{model_name}: Strata počas trénovania')
    ax1.set_xlabel('Epocha')
    ax1.set_ylabel('Strata')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy alebo iná metrika
    ax2 = axes[1]
    metric_keys = [k for k in history.keys() if 'loss' not in k and 'val_' not in k]
    
    for key in metric_keys:
        ax2.plot(history[key], label=f'Trénovanie - {key}', color=config.COLORS['train'])
        val_key = f'val_{key}'
        if val_key in history:
            ax2.plot(history[val_key], label=f'Validácia - {key}', color=config.COLORS['validation'])
    
    ax2.set_title(f'{model_name}: Metrika počas trénovania')
    ax2.set_xlabel('Epocha')
    ax2.set_ylabel('Hodnota')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


# =============================================================================
# SIMULÁCIA A BACKTESTING
# =============================================================================

def plot_monte_carlo_simulation(
    simulations: np.ndarray,
    initial_price: float,
    ticker: str = None,
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí výsledky Monte Carlo simulácie.
    
    Args:
        simulations (np.ndarray): Matica simulácií (n_simulations x n_days)
        initial_price (float): Počiatočná cena
        ticker (str, optional): Symbol akcie
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Simulované cesty
    ax1 = axes[0]
    n_simulations = simulations.shape[0]
    
    # Zobraz max 100 ciest pre prehľadnosť
    n_show = min(100, n_simulations)
    for i in range(n_show):
        ax1.plot(simulations[i], alpha=0.1, color=config.COLORS['primary'])
    
    # Percentily
    percentiles = [5, 50, 95]
    for p in percentiles:
        ax1.plot(np.percentile(simulations, p, axis=0), 
                linewidth=2, label=f'{p}. percentil')
    
    ax1.axhline(y=initial_price, color='black', linestyle='--', alpha=0.5, label='Počiatočná cena')
    ax1.set_title(f'Monte Carlo simulácia{" - " + ticker if ticker else ""}')
    ax1.set_xlabel('Dni')
    ax1.set_ylabel('Cena')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribúcia koncových cien
    ax2 = axes[1]
    final_prices = simulations[:, -1]
    ax2.hist(final_prices, bins=50, color=config.COLORS['primary'], 
             alpha=0.7, edgecolor='black', density=True)
    ax2.axvline(x=initial_price, color='black', linestyle='--', label='Počiatočná cena')
    ax2.axvline(x=np.mean(final_prices), color='red', linestyle='-', label=f'Priemer: {np.mean(final_prices):.2f}')
    ax2.set_title('Distribúcia koncových cien')
    ax2.set_xlabel('Cena')
    ax2.set_ylabel('Hustota')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


def plot_simulation_returns_and_prices(
    simulations: np.ndarray,
    initial_price: float,
    ticker: str = None,
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí simulované výnosy aj konečné ceny v jednom prehľadnom grafe.

    Obsahuje:
    - Distribúciu simulovaných ročných výnosov (%)
    - Distribúciu konečných cien
    - Vývoj simulovaných kumulatívnych výnosov v čase
    - Súhrnnú tabuľku kľúčových štatistík

    Args:
        simulations (np.ndarray): Matica simulácií (n_simulations x n_days+1)
        initial_price (float): Počiatočná cena
        ticker (str, optional): Symbol akcie
        save_path (str, optional): Cesta pre uloženie

    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    final_prices = simulations[:, -1]
    total_returns_pct = (final_prices / initial_price - 1) * 100

    # --- 1. Distribúcia ročných výnosov ---
    ax1 = axes[0, 0]
    ax1.hist(total_returns_pct, bins=80, density=True, alpha=0.7,
             color=config.COLORS['primary'], edgecolor='none')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.6, label='0 %')
    ax1.axvline(x=np.mean(total_returns_pct), color='red', linestyle='-',
                linewidth=2, label=f'Priemer: {np.mean(total_returns_pct):.1f} %')
    ax1.axvline(x=np.median(total_returns_pct), color='green', linestyle='-',
                linewidth=2, label=f'Medián: {np.median(total_returns_pct):.1f} %')
    ax1.set_title('Distribúcia simulovaných výnosov')
    ax1.set_xlabel('Celkový výnos (%)')
    ax1.set_ylabel('Hustota')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- 2. Distribúcia konečných cien ---
    ax2 = axes[0, 1]
    ax2.hist(final_prices, bins=80, density=True, alpha=0.7,
             color=config.COLORS['secondary'], edgecolor='none')
    ax2.axvline(x=initial_price, color='black', linestyle='--', linewidth=1.5,
                label=f'Počiatočná: {initial_price:.2f}')
    ax2.axvline(x=np.mean(final_prices), color='red', linestyle='-',
                linewidth=2, label=f'Priemer: {np.mean(final_prices):.2f}')
    ax2.axvline(x=np.median(final_prices), color='green', linestyle='-',
                linewidth=2, label=f'Medián: {np.median(final_prices):.2f}')
    ax2.set_title('Distribúcia konečných cien')
    ax2.set_xlabel('Cena')
    ax2.set_ylabel('Hustota')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- 3. Kumulatívne výnosy v čase (percentily) ---
    ax3 = axes[1, 0]
    n_days = simulations.shape[1]
    cum_returns = (simulations / initial_price - 1) * 100

    n_show = min(80, simulations.shape[0])
    for i in range(n_show):
        ax3.plot(cum_returns[i], alpha=0.05, color=config.COLORS['primary'], linewidth=0.5)

    for p, col, lbl in [(5, '#d62728', '5. percentil'),
                         (50, '#2ca02c', 'Medián'),
                         (95, '#1f77b4', '95. percentil')]:
        ax3.plot(np.percentile(cum_returns, p, axis=0),
                 linewidth=2, color=col, label=lbl)

    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Vývoj simulovaných výnosov v čase')
    ax3.set_xlabel('Dni')
    ax3.set_ylabel('Kumulatívny výnos (%)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # --- 4. Súhrnná tabuľka ---
    ax4 = axes[1, 1]
    ax4.axis('off')

    prob_growth = (final_prices > initial_price).mean() * 100
    rows = [
        ['Počiatočná cena', f'{initial_price:.2f}'],
        ['Priemerná konečná cena', f'{np.mean(final_prices):.2f}'],
        ['Medián konečnej ceny', f'{np.median(final_prices):.2f}'],
        ['5. percentil ceny', f'{np.percentile(final_prices, 5):.2f}'],
        ['95. percentil ceny', f'{np.percentile(final_prices, 95):.2f}'],
        ['Priemerný výnos', f'{np.mean(total_returns_pct):.2f} %'],
        ['Medián výnosu', f'{np.median(total_returns_pct):.2f} %'],
        ['Volatilita výnosu', f'{np.std(total_returns_pct):.2f} %'],
        ['P(rast)', f'{prob_growth:.1f} %'],
        ['Min. konečná cena', f'{np.min(final_prices):.2f}'],
        ['Max. konečná cena', f'{np.max(final_prices):.2f}'],
    ]

    tbl = ax4.table(
        cellText=rows,
        colLabels=['Metrika', 'Hodnota'],
        loc='center', cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#F2F2F2')
    ax4.set_title('Súhrnné štatistiky simulácie', pad=8)

    title = f'Simulované výnosy a konečné ceny{" - " + ticker if ticker else ""}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")

    return fig


def plot_backtest_results(
    backtest_results: Dict,
    save_path: str = None
) -> plt.Figure:
    """
    Vykreslí výsledky backtestingu.
    
    Args:
        backtest_results (Dict): Výsledky backtestingu
        save_path (str, optional): Cesta pre uloženie
        
    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Equity krivka
    ax1 = axes[0, 0]
    if 'equity_curve' in backtest_results:
        equity = backtest_results['equity_curve']
        ax1.plot(equity, color=config.COLORS['primary'], linewidth=1.5)
        ax1.set_title('Vývoj portfólia')
        ax1.set_xlabel('Obchod')
        ax1.set_ylabel('Hodnota portfólia')
        ax1.grid(True, alpha=0.3)
    
    # Kumulatívne výnosy
    ax2 = axes[0, 1]
    if 'cumulative_returns' in backtest_results:
        cum_returns = backtest_results['cumulative_returns']
        ax2.plot(cum_returns, color=config.COLORS['primary'], linewidth=1.5)
        ax2.set_title('Kumulatívne výnosy')
        ax2.set_xlabel('Čas')
        ax2.set_ylabel('Kumulatívny výnos (%)')
        ax2.grid(True, alpha=0.3)
    
    # Distribúcia výnosov z obchodov
    ax3 = axes[1, 0]
    if 'trade_returns' in backtest_results:
        trade_returns = backtest_results['trade_returns']
        colors = [config.COLORS['positive'] if r >= 0 else config.COLORS['negative'] for r in trade_returns]
        ax3.bar(range(len(trade_returns)), trade_returns, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Výnosy jednotlivých obchodov')
        ax3.set_xlabel('Obchod')
        ax3.set_ylabel('Výnos (%)')
        ax3.grid(True, alpha=0.3)
    
    # Štatistiky
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if 'metrics' in backtest_results:
        metrics = backtest_results['metrics']
        text = "ŠTATISTIKY BACKTESTINGU\n" + "="*30 + "\n\n"
        for key, value in metrics.items():
            if isinstance(value, float):
                text += f"{key}: {value:.4f}\n"
            else:
                text += f"{key}: {value}\n"
        
        ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")
    
    return fig


# =============================================================================
# DISTRIBUČNÉ FITOVANIE - HYBRIDNÝ PRÍSTUP
# =============================================================================

def plot_distribution_fit(
    returns: np.ndarray,
    fitter,
    ticker: str = None,
    save_path: str = None
) -> list:
    """
    Vykreslí 3 samostatné grafy pre fitovanie rozdelenia:
      1) Histogram empirických log-výnosov + nafitované hustoty
      2) Q-Q graf empirických vs. teoretických kvantilov
      3) Tabuľka výsledkov AIC / KS testu

    Ak je zadaný save_path, každý graf sa uloží ako samostatný súbor:
      save_path_hist.png, save_path_qq.png, save_path_table.png

    Args:
        returns (np.ndarray): Log-výnosy
        fitter: Nafitovaný ReturnDistributionFitter
        ticker (str, optional): Symbol akcie
        save_path (str, optional): Cesta pre uloženie (bez prípony sa
            použije ako základ pre 3 súbory)

    Returns:
        list[plt.Figure]: Zoznam troch figure objektov
    """
    from scipy import stats as scipy_stats

    title_suffix = f' – {ticker}' if ticker else ''
    figures = []

    # Spoločné prípravy
    x_min, x_max = np.percentile(returns, 0.5), np.percentile(returns, 99.5)
    x_grid = np.linspace(x_min, x_max, 400)
    palette = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b',
               '#e377c2', '#7f7f7f', '#bcbd22']
    sorted_dists = sorted(
        fitter.all_results.items(),
        key=lambda item: item[1]['aic']
    )

    # Základ cesty pre ukladanie
    if save_path:
        base, ext = os.path.splitext(save_path)
        if not ext:
            ext = '.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ── 1. Histogram + nafitované hustoty ──
    fig_hist, ax_hist = plt.subplots(figsize=(12, 7))
    ax_hist.hist(
        returns, bins=80, density=True, alpha=0.55,
        color=config.COLORS['primary'], edgecolor='none', label='Empirické výnosy'
    )

    for (dist_name, res), color in zip(sorted_dists, palette):
        try:
            dist = getattr(scipy_stats, dist_name)
            pdf_vals = dist.pdf(x_grid, *res['params'])
            is_best  = (dist_name == fitter.best_distribution)
            ax_hist.plot(
                x_grid, pdf_vals,
                color=color,
                linestyle='-' if is_best else '--',
                linewidth=2.5 if is_best else 1.0,
                label=f"{res['name_sk']} (AIC: {res['aic']:.0f})"
            )
        except Exception:
            pass

    ax_hist.set_title(f'Fitovanie rozdelenia log-výnosov{title_suffix}', fontsize=13)
    ax_hist.set_xlabel('Log-výnosy')
    ax_hist.set_ylabel('Hustota pravdepodobnosti')
    ax_hist.legend(fontsize=10, loc='upper right')
    ax_hist.grid(True, alpha=0.3)
    fig_hist.tight_layout()

    if save_path:
        fig_hist.savefig(f"{base}_hist{ext}", dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {base}_hist{ext}")
    figures.append(fig_hist)

    # ── 2. Q-Q graf ──
    fig_qq, ax_qq = plt.subplots(figsize=(8, 7))
    if fitter.fit_complete:
        dist_name = fitter.best_distribution
        dist_obj  = getattr(scipy_stats, dist_name)
        n = len(returns)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical_q = dist_obj.ppf(probs, *fitter.best_params)
        empirical_q   = np.sort(returns)

        mask = np.isfinite(theoretical_q)
        ax_qq.scatter(
            theoretical_q[mask], empirical_q[mask],
            alpha=0.4, s=6, color=config.COLORS['primary']
        )
        lim = [max(x_min, theoretical_q[mask].min()),
               min(x_max, theoretical_q[mask].max())]
        ax_qq.plot(lim, lim, 'r-', linewidth=1.5, label='Ideálna zhoda')
        ax_qq.set_title(
            f'Q-Q: {fitter.all_results[dist_name]["name_sk"]}{title_suffix}',
            fontsize=13
        )
        ax_qq.set_xlabel('Teoretické kvantily')
        ax_qq.set_ylabel('Empirické kvantily')
        ax_qq.legend(fontsize=11)
        ax_qq.grid(True, alpha=0.3)
    fig_qq.tight_layout()

    if save_path:
        fig_qq.savefig(f"{base}_qq{ext}", dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {base}_qq{ext}")
    figures.append(fig_qq)

    # ── 3. Tabuľka výsledkov ──
    n_rows = len(sorted_dists)
    fig_tbl, ax_table = plt.subplots(figsize=(12, 1.2 + n_rows * 0.45))
    ax_table.axis('off')

    moments = fitter.get_empirical_moments()
    header = ['Rozdelenie', 'AIC', 'BIC', 'KS-štatistika', 'KS p-hodnota', 'Najlepšie?']
    rows = []
    for dist_name, res in sorted_dists:
        rows.append([
            res['name_sk'],
            f"{res['aic']:.2f}",
            f"{res['bic']:.2f}",
            f"{res['ks_statistic']:.4f}",
            f"{res['ks_pvalue']:.4f}",
            '✓ ÁNO' if dist_name == fitter.best_distribution else '',
        ])

    tbl = ax_table.table(
        cellText=rows, colLabels=header,
        loc='center', cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(list(range(len(header))))
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif rows[r - 1][-1].startswith('✓'):
            cell.set_facecolor('#E2EFDA')
    ax_table.set_title(
        f'Výsledky fitovania{title_suffix}  –  '
        f'μ={moments["mean"]:.5f}, σ={moments["std"]:.5f}, '
        f'šikmosť={moments["skewness"]:.3f}, špicatosť={moments["kurtosis"]:.3f}',
        fontsize=11, pad=10
    )
    fig_tbl.tight_layout()

    if save_path:
        fig_tbl.savefig(f"{base}_table{ext}", dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {base}_table{ext}")
    figures.append(fig_tbl)

    return figures


def plot_mc_comparison(
    gbm_simulations: np.ndarray,
    fitted_simulations: np.ndarray,
    initial_price: float,
    dist_name: str = 'fitted',
    ticker: str = None,
    save_path: str = None
) -> plt.Figure:
    """
    Porovnáva GBM simuláciu (normálne rozdelenie) s distribučne
    nafitovanou simuláciou.

    Obsahuje:
    - Cesty GBM simulácie s percentilmi
    - Cesty nafitovanej MC simulácie s percentilmi
    - Porovnanie distribucí konečných cien
    - Porovnanie rizikových metrík (VaR 95%, 99%, priemer)

    Args:
        gbm_simulations (np.ndarray): Výsledky GBM MC simulácie
        fitted_simulations (np.ndarray): Výsledky nafitovanej MC simulácie
        initial_price (float): Počiatočná cena
        dist_name (str): Názov nafitovanej distribúcie
        ticker (str, optional): Symbol akcie
        save_path (str, optional): Cesta pre uloženie

    Returns:
        plt.Figure: Matplotlib figure objekt
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    n_show = min(80, gbm_simulations.shape[0])
    percentile_levels = [5, 50, 95]
    pct_colors = ['#d62728', '#2ca02c', '#1f77b4']

    # --- GBM cesty ---
    ax1 = axes[0, 0]
    for i in range(n_show):
        ax1.plot(gbm_simulations[i], alpha=0.06, color=config.COLORS['primary'], linewidth=0.5)
    for p, col in zip(percentile_levels, pct_colors):
        ax1.plot(
            np.percentile(gbm_simulations, p, axis=0),
            linewidth=2, color=col, label=f'{p}. percentil'
        )
    ax1.axhline(y=initial_price, color='black', linestyle='--', alpha=0.6, label='Počiatočná cena')
    ax1.set_title('GBM (normálne rozdelenie)')
    ax1.set_xlabel('Dni')
    ax1.set_ylabel('Cena')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Nafitované rozdelenie cesty ---
    ax2 = axes[0, 1]
    for i in range(n_show):
        ax2.plot(fitted_simulations[i], alpha=0.06, color=config.COLORS['secondary'], linewidth=0.5)
    for p, col in zip(percentile_levels, pct_colors):
        ax2.plot(
            np.percentile(fitted_simulations, p, axis=0),
            linewidth=2, color=col, label=f'{p}. percentil'
        )
    ax2.axhline(y=initial_price, color='black', linestyle='--', alpha=0.6, label='Počiatočná cena')
    ax2.set_title(f'Nafitované rozdelenie ({dist_name})')
    ax2.set_xlabel('Dni')
    ax2.set_ylabel('Cena')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Porovnanie distribucí konečných cien ---
    ax3 = axes[1, 0]
    gbm_final    = gbm_simulations[:, -1]
    fitted_final = fitted_simulations[:, -1]
    ax3.hist(
        gbm_final, bins=60, density=True, alpha=0.5,
        color=config.COLORS['primary'], label='GBM', edgecolor='none'
    )
    ax3.hist(
        fitted_final, bins=60, density=True, alpha=0.5,
        color=config.COLORS['secondary'], label=f'Nafitované ({dist_name})', edgecolor='none'
    )
    ax3.axvline(x=initial_price, color='black', linestyle='--', linewidth=1.5, label='Počiatočná cena')
    ax3.set_title('Porovnanie distribúcií konečných cien')
    ax3.set_xlabel('Cena po simulácii')
    ax3.set_ylabel('Hustota')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # --- Porovnanie metrík ---
    ax4 = axes[1, 1]
    ax4.axis('off')

    def _pct_return(arr):
        return (arr / initial_price - 1) * 100

    gbm_ret    = _pct_return(gbm_final)
    fitted_ret = _pct_return(fitted_final)

    def _var(ret, cl):
        return -np.percentile(ret, (1 - cl) * 100)

    def _cvar(ret, var_val):
        losses = ret[ret <= -var_val]
        return -np.mean(losses) if len(losses) > 0 else var_val

    rows = [
        ['Metrika', 'GBM', f'Nafitované ({dist_name})'],
        ['Priem. výnos (%)',
         f"{np.mean(gbm_ret):.2f}", f"{np.mean(fitted_ret):.2f}"],
        ['Medián výnosu (%)',
         f"{np.median(gbm_ret):.2f}", f"{np.median(fitted_ret):.2f}"],
        ['Std výnosu (%)',
         f"{np.std(gbm_ret):.2f}", f"{np.std(fitted_ret):.2f}"],
        ['P(rast)',
         f"{(gbm_ret > 0).mean()*100:.1f}%", f"{(fitted_ret > 0).mean()*100:.1f}%"],
        ['VaR 95% (%)',
         f"{_var(gbm_ret, 0.95):.2f}", f"{_var(fitted_ret, 0.95):.2f}"],
        ['CVaR 95% (%)',
         f"{_cvar(gbm_ret, _var(gbm_ret, 0.95)):.2f}",
         f"{_cvar(fitted_ret, _var(fitted_ret, 0.95)):.2f}"],
        ['VaR 99% (%)',
         f"{_var(gbm_ret, 0.99):.2f}", f"{_var(fitted_ret, 0.99):.2f}"],
        ['CVaR 99% (%)',
         f"{_cvar(gbm_ret, _var(gbm_ret, 0.99)):.2f}",
         f"{_cvar(fitted_ret, _var(fitted_ret, 0.99)):.2f}"],
        ['5. percentil ceny',
         f"{np.percentile(gbm_final, 5):.2f}", f"{np.percentile(fitted_final, 5):.2f}"],
        ['95. percentil ceny',
         f"{np.percentile(gbm_final, 95):.2f}", f"{np.percentile(fitted_final, 95):.2f}"],
    ]

    tbl = ax4.table(
        cellText=[r for r in rows[1:]],
        colLabels=rows[0],
        loc='center', cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#F2F2F2')
    ax4.set_title('Porovnanie rizikových metrík', pad=8)

    title = f'Porovnanie MC metód{" - " + ticker if ticker else ""}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[OK] Graf uložený do {save_path}")

    return fig


# =============================================================================
# POMOCNÉ FUNKCIE
# =============================================================================

def create_visualization_report(
    df: pd.DataFrame,
    ticker: str,
    output_dir: str = None
) -> None:
    """
    Vytvorí kompletný vizualizačný report pre dané dáta.
    
    Args:
        df (pd.DataFrame): DataFrame s dátami a features
        ticker (str): Symbol akcie
        output_dir (str, optional): Adresár pre uloženie grafov
    """
    if output_dir is None:
        output_dir = os.path.join(config.PLOTS_DIR, ticker)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"VYTVÁRANIE VIZUALIZAČNÉHO REPORTU PRE {ticker}")
    print("=" * 60)
    
    # 1. Cenová história
    print("[INFO] Vytváranie cenového grafu...")
    plot_price_history(df, ticker, save_path=os.path.join(output_dir, "01_price_history.png"))
    plt.close()
    
    # 2. Technické indikátory
    print("[INFO] Vytváranie grafu technických indikátorov...")
    plot_technical_indicators(df, ticker, save_path=os.path.join(output_dir, "02_technical_indicators.png"))
    plt.close()
    
    # 3. Bollinger Bands
    if 'BB_Upper' in df.columns:
        print("[INFO] Vytváranie Bollinger Bands grafu...")
        plot_bollinger_bands(df, ticker, save_path=os.path.join(output_dir, "03_bollinger_bands.png"))
        plt.close()
    
    # 4. Distribúcia výnosov
    if 'Return_1d' in df.columns:
        print("[INFO] Vytváranie grafu distribúcie výnosov...")
        plot_returns_distribution(df, 'Return_1d', ticker, save_path=os.path.join(output_dir, "04_returns_distribution.png"))
        plt.close()
    
    # 5. Korelačná matica
    print("[INFO] Vytváranie korelačnej matice...")
    key_columns = ['Close', 'Volume', 'RSI', 'MACD', 'ATR', 'Return_1d', 'Volatility_20d']
    existing_cols = [c for c in key_columns if c in df.columns]
    if len(existing_cols) > 1:
        plot_correlation_matrix(df, existing_cols, 
                               title=f"Korelačná matica - {ticker}",
                               save_path=os.path.join(output_dir, "05_correlation_matrix.png"))
        plt.close()
    
    print(f"\n[OK] Vizualizačný report uložený do {output_dir}")
    print("=" * 60)


# =============================================================================
# HLAVNÁ FUNKCIA PRE TESTOVANIE MODULU
# =============================================================================

if __name__ == "__main__":
    import data_downloader
    import data_preprocessing
    import feature_engineering
    
    print("=" * 60)
    print("TESTOVANIE MODULU VISUALIZATION")
    print("=" * 60)
    
    # Načítanie dát
    try:
        df = data_downloader.load_stock_data("AAPL")
    except FileNotFoundError:
        print("[INFO] Dáta nie sú k dispozícii, sťahujem...")
        df = data_downloader.download_stock_data("AAPL")
    
    # Predspracovanie a features
    df_processed = data_preprocessing.preprocess_pipeline(df, ticker="AAPL", save_to_file=False)
    df_features = feature_engineering.create_all_features(df_processed, ticker="AAPL", save_to_file=False)
    
    # Vytvorenie reportu
    create_visualization_report(df_features, "AAPL")
    
    print("\n[OK] Testovanie vizualizácie dokončené")
