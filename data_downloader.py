# -*- coding: utf-8 -*-
"""
Modul pre sťahovanie finančných dát z Yahoo Finance.

Tento modul poskytuje funkcie na:
- Sťahovanie historických dát pre jednotlivé akcie
- Sťahovanie dát pre viacero akcií naraz
- Ukladanie a načítavanie dát z lokálnych súborov
- Získanie základných informácií o akciách

Autor: Dominika Melicherová
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import List, Dict, Optional, Union

# Import konfigurácie
import config


def download_stock_data(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    save_to_file: bool = True
) -> pd.DataFrame:
    """
    Stiahne historické dáta pre zadaný akciový ticker z Yahoo Finance.
    
    Táto funkcia sťahuje OHLCV (Open, High, Low, Close, Volume) dáta
    pre zadanú akciu za určené časové obdobie.
    
    Args:
        ticker (str): Symbol akcie (napr. "AAPL", "MSFT")
        start_date (str, optional): Počiatočný dátum vo formáte "YYYY-MM-DD".
                                    Predvolené: hodnota z config.py
        end_date (str, optional): Koncový dátum vo formáte "YYYY-MM-DD".
                                  Predvolené: hodnota z config.py
        save_to_file (bool): Ak True, uloží dáta do CSV súboru
        
    Returns:
        pd.DataFrame: DataFrame s historickými dátami akcie
        
    Raises:
        ValueError: Ak sa nepodarí stiahnuť dáta pre zadaný ticker
        
    Example:
        >>> df = download_stock_data("AAPL", "2020-01-01", "2024-01-01")
        >>> print(df.head())
    """
    # Použitie predvolených hodnôt z konfigurácie ak nie sú zadané
    if start_date is None:
        start_date = config.START_DATE
    if end_date is None:
        end_date = config.END_DATE
    
    print(f"[INFO] Sťahujem dáta pre {ticker} od {start_date} do {end_date}...")
    
    try:
        # Vytvorenie objektu Ticker
        stock = yf.Ticker(ticker)
        
        # Stiahnutie historických dát
        df = stock.history(start=start_date, end=end_date)
        
        # Kontrola či boli dáta stiahnuté
        if df.empty:
            raise ValueError(f"Nepodarilo sa stiahnuť dáta pre ticker {ticker}")
        
        # Resetovanie indexu - dátum bude ako stĺpec
        df = df.reset_index()
        
        # Premenovanie stĺpcov na slovenské názvy pre lepšiu čitateľnosť
        # Ponechávame aj anglické názvy pre kompatibilitu
        df.columns = [col if col != 'Date' else 'Datum' for col in df.columns]
        
        # Pridanie stĺpca s tickerom
        df['Ticker'] = ticker
        
        # Odstránenie stĺpcov Dividends a Stock Splits ak existujú (nie sú potrebné)
        columns_to_drop = ['Dividends', 'Stock Splits']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        print(f"[OK] Úspešne stiahnutých {len(df)} záznamov pre {ticker}")
        
        # Uloženie do súboru ak je požadované
        if save_to_file:
            filepath = os.path.join(config.DATA_DIR, f"{ticker}_raw.csv")
            df.to_csv(filepath, index=False)
            print(f"[OK] Dáta uložené do {filepath}")
        
        return df
        
    except Exception as e:
        print(f"[CHYBA] Nepodarilo sa stiahnuť dáta pre {ticker}: {str(e)}")
        raise ValueError(f"Chyba pri sťahovaní dát pre {ticker}: {str(e)}")


def download_multiple_stocks(
    tickers: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    save_to_file: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Stiahne historické dáta pre viacero akciových tickerov.
    
    Args:
        tickers (List[str], optional): Zoznam tickerov na stiahnutie.
                                       Predvolené: zoznam z config.py
        start_date (str, optional): Počiatočný dátum
        end_date (str, optional): Koncový dátum
        save_to_file (bool): Ak True, uloží každý ticker do samostatného súboru
        
    Returns:
        Dict[str, pd.DataFrame]: Slovník kde kľúče sú tickery a hodnoty sú DataFrames
        
    Example:
        >>> data = download_multiple_stocks(["AAPL", "MSFT", "GOOGL"])
        >>> print(data.keys())
    """
    # Použitie predvolených tickerov z konfigurácie
    if tickers is None:
        tickers = config.get_ticker_list()
    
    print("=" * 60)
    print(f"SŤAHOVANIE DÁT PRE {len(tickers)} TICKEROV")
    print("=" * 60)
    
    # Slovník pre uloženie dát
    all_data = {}
    
    # Počítadlo úspešných a neúspešných stiahnutí
    success_count = 0
    fail_count = 0
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Spracovávam {ticker}...")
        
        try:
            df = download_stock_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                save_to_file=save_to_file
            )
            all_data[ticker] = df
            success_count += 1
            
        except ValueError as e:
            print(f"[VAROVANIE] Preskakujem {ticker} kvôli chybe")
            fail_count += 1
            continue
    
    # Súhrnná správa
    print("\n" + "=" * 60)
    print("SÚHRN SŤAHOVANIA")
    print("=" * 60)
    print(f"Úspešne stiahnuté: {success_count}/{len(tickers)}")
    print(f"Neúspešné: {fail_count}/{len(tickers)}")
    
    if all_data:
        # Uloženie kombinovaných dát
        if save_to_file:
            combined_df = pd.concat(all_data.values(), ignore_index=True)
            combined_filepath = os.path.join(config.DATA_DIR, "all_stocks_raw.csv")
            combined_df.to_csv(combined_filepath, index=False)
            print(f"[OK] Kombinované dáta uložené do {combined_filepath}")
    
    return all_data


def load_stock_data(ticker: str) -> pd.DataFrame:
    """
    Načíta dáta akcie z lokálneho CSV súboru.
    
    Táto funkcia slúži na načítanie predtým stiahnutých dát
    bez potreby opätovného sťahovania z internetu.
    
    Args:
        ticker (str): Symbol akcie
        
    Returns:
        pd.DataFrame: DataFrame s dátami akcie
        
    Raises:
        FileNotFoundError: Ak súbor neexistuje
    """
    filepath = os.path.join(config.DATA_DIR, f"{ticker}_raw.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Súbor {filepath} neexistuje. "
            f"Najprv stiahnite dáta pomocou download_stock_data('{ticker}')"
        )
    
    print(f"[INFO] Načítavam dáta pre {ticker} z {filepath}...")
    df = pd.read_csv(filepath, parse_dates=['Datum'])
    print(f"[OK] Načítaných {len(df)} záznamov")
    
    return df


def load_all_stocks_data() -> pd.DataFrame:
    """
    Načíta kombinované dáta všetkých akcií z lokálneho súboru.
    
    Returns:
        pd.DataFrame: DataFrame s dátami všetkých akcií
    """
    filepath = os.path.join(config.DATA_DIR, "all_stocks_raw.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Súbor {filepath} neexistuje. "
            f"Najprv stiahnite dáta pomocou download_multiple_stocks()"
        )
    
    print(f"[INFO] Načítavam kombinované dáta z {filepath}...")
    df = pd.read_csv(filepath, parse_dates=['Datum'])
    print(f"[OK] Načítaných {len(df)} záznamov pre {df['Ticker'].nunique()} tickerov")
    
    return df


def get_stock_info(ticker: str) -> Dict:
    """
    Získa základné informácie o akcii.
    
    Args:
        ticker (str): Symbol akcie
        
    Returns:
        Dict: Slovník s informáciami o akcii (názov spoločnosti, sektor, atď.)
    """
    print(f"[INFO] Získavam informácie o {ticker}...")
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Výber relevantných informácií
        relevant_info = {
            'ticker': ticker,
            'nazov': info.get('longName', 'N/A'),
            'sektor': info.get('sector', 'N/A'),
            'odvetvie': info.get('industry', 'N/A'),
            'krajina': info.get('country', 'N/A'),
            'mena': info.get('currency', 'N/A'),
            'trhova_kapitalizacia': info.get('marketCap', 'N/A'),
            'priemerny_objem': info.get('averageVolume', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            '52_tyzdnove_maximum': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52_tyzdnove_minimum': info.get('fiftyTwoWeekLow', 'N/A'),
            'dividendovy_vynos': info.get('dividendYield', 'N/A'),
        }
        
        print(f"[OK] Informácie o {ticker} získané")
        return relevant_info
        
    except Exception as e:
        print(f"[CHYBA] Nepodarilo sa získať informácie o {ticker}: {str(e)}")
        return {'ticker': ticker, 'chyba': str(e)}


def get_multiple_stocks_info(tickers: List[str] = None) -> pd.DataFrame:
    """
    Získa informácie o viacerých akciách.
    
    Args:
        tickers (List[str], optional): Zoznam tickerov
        
    Returns:
        pd.DataFrame: DataFrame s informáciami o akciách
    """
    if tickers is None:
        tickers = config.get_ticker_list()
    
    print("=" * 60)
    print(f"ZÍSKAVANIE INFORMÁCIÍ PRE {len(tickers)} TICKEROV")
    print("=" * 60)
    
    all_info = []
    
    for ticker in tickers:
        info = get_stock_info(ticker)
        all_info.append(info)
    
    df = pd.DataFrame(all_info)
    
    # Uloženie do súboru
    filepath = os.path.join(config.DATA_DIR, "stocks_info.csv")
    df.to_csv(filepath, index=False)
    print(f"\n[OK] Informácie uložené do {filepath}")
    
    return df


def check_data_availability(ticker: str) -> Dict:
    """
    Skontroluje dostupnosť dát pre daný ticker.
    
    Args:
        ticker (str): Symbol akcie
        
    Returns:
        Dict: Informácie o dostupnosti dát
    """
    raw_filepath = os.path.join(config.DATA_DIR, f"{ticker}_raw.csv")
    processed_filepath = os.path.join(config.DATA_DIR, f"{ticker}_processed.csv")
    
    availability = {
        'ticker': ticker,
        'raw_data_exists': os.path.exists(raw_filepath),
        'processed_data_exists': os.path.exists(processed_filepath),
    }
    
    if availability['raw_data_exists']:
        df = pd.read_csv(raw_filepath)
        availability['raw_records'] = len(df)
        availability['raw_date_range'] = f"{df['Datum'].min()} - {df['Datum'].max()}"
    
    return availability


def print_data_summary(df: pd.DataFrame, ticker: str = None) -> None:
    """
    Vypíše súhrn dát pre danú akciu.
    
    Args:
        df (pd.DataFrame): DataFrame s dátami
        ticker (str, optional): Symbol akcie pre zobrazenie v nadpise
    """
    title = f"SÚHRN DÁT" + (f" PRE {ticker}" if ticker else "")
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    print(f"\nPočet záznamov: {len(df)}")
    
    if 'Datum' in df.columns:
        print(f"Časové obdobie: {df['Datum'].min()} až {df['Datum'].max()}")
    
    print(f"\nStĺpce: {', '.join(df.columns.tolist())}")
    
    print("\nZákladná štatistika:")
    print(df.describe().round(2))
    
    print("\nChýbajúce hodnoty:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("Žiadne chýbajúce hodnoty")
    
    print("=" * 60)


# =============================================================================
# HLAVNÁ FUNKCIA PRE TESTOVANIE MODULU
# =============================================================================

if __name__ == "__main__":
    # Testovanie funkcií modulu
    print("=" * 60)
    print("TESTOVANIE MODULU DATA_DOWNLOADER")
    print("=" * 60)
    
    # Test stiahnutia dát pre jeden ticker
    print("\n1. Test stiahnutia dát pre jeden ticker (AAPL):")
    try:
        df = download_stock_data("AAPL")
        print_data_summary(df, "AAPL")
    except Exception as e:
        print(f"Chyba: {e}")
    
    # Test získania informácií o akcii
    print("\n2. Test získania informácií o akcii:")
    info = get_stock_info("AAPL")
    for key, value in info.items():
        print(f"  {key}: {value}")
