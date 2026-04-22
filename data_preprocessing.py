# -*- coding: utf-8 -*-
"""
Modul pre predspracovanie finančných dát.

Tento modul poskytuje funkcie na:
- Čistenie dát a spracovanie chýbajúcich hodnôt
- Detekciu a ošetrenie odľahlých hodnôt (outliers)
- Normalizáciu a štandardizáciu dát
- Vytváranie cieľových premenných pre predikciu
- Rozdelenie dát na trénovaciu, validačnú a testovaciu množinu

Autor: Dominika Melicherová
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Import konfigurácie
import config


class DataPreprocessor:
    """
    Trieda pre komplexné predspracovanie finančných dát.
    
    Táto trieda zapuzdruje všetky operácie predspracovania vrátane
    čistenia dát, detekcie outlierov, normalizácie a vytvárania
    cieľových premenných.
    
    Attributes:
        scaler: Objekt sklearn scaler pre normalizáciu
        scaler_type (str): Typ použitého scalera
        fitted (bool): Indikátor či bol scaler natrenovaný
    """
    
    def __init__(self, scaler_type: str = "standard"):
        """
        Inicializácia predprocesora.
        
        Args:
            scaler_type (str): Typ scalera - "standard", "minmax", alebo "robust"
        """
        self.scaler_type = scaler_type
        self.scaler = self._create_scaler(scaler_type)
        self.fitted = False
        self.feature_columns = None
        
    def _create_scaler(self, scaler_type: str):
        """
        Vytvorí scaler podľa zadaného typu.
        
        Args:
            scaler_type (str): Typ scalera
            
        Returns:
            sklearn scaler objekt
        """
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler()
        }
        
        if scaler_type not in scalers:
            print(f"[VAROVANIE] Neznámy typ scalera '{scaler_type}'. Použije sa StandardScaler.")
            return StandardScaler()
        
        return scalers[scaler_type]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vyčistí dáta - odstráni duplikáty a spracuje chýbajúce hodnoty.
    
    Args:
        df (pd.DataFrame): Vstupný DataFrame
        
    Returns:
        pd.DataFrame: Vyčistený DataFrame
    """
    print("[INFO] Čistenie dát...")
    
    original_len = len(df)
    
    # Kópia dát
    df_clean = df.copy()
    
    # Odstránenie duplikátov
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = original_len - len(df_clean)
    if duplicates_removed > 0:
        print(f"  - Odstránených {duplicates_removed} duplikátov")
    
    # Kontrola chýbajúcich hodnôt
    missing_before = df_clean.isnull().sum().sum()
    
    if missing_before > 0:
        print(f"  - Nájdených {missing_before} chýbajúcich hodnôt")
        
        # Stratégia pre spracovanie chýbajúcich hodnôt:
        # 1. Pre numerické stĺpce - forward fill, potom backward fill
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(method='ffill')
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(method='bfill')
        
        # 2. Odstránenie riadkov ktoré stále obsahujú NaN
        df_clean = df_clean.dropna()
        
        missing_after = df_clean.isnull().sum().sum()
        print(f"  - Po čistení zostáva {missing_after} chýbajúcich hodnôt")
    else:
        print("  - Žiadne chýbajúce hodnoty")
    
    rows_removed = original_len - len(df_clean)
    print(f"[OK] Čistenie dokončené. Odstránených {rows_removed} riadkov ({rows_removed/original_len*100:.2f}%)")
    
    return df_clean


def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: List[str] = None,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detekuje odľahlé hodnoty pomocou Z-score metódy.
    
    Z-score meria koľko štandardných odchýlok je hodnota vzdialená od priemeru.
    Hodnoty s |Z-score| > threshold sú považované za odľahlé.
    
    Args:
        df (pd.DataFrame): Vstupný DataFrame
        columns (List[str], optional): Stĺpce na analýzu. Ak None, použijú sa všetky numerické.
        threshold (float): Prah pre identifikáciu outlierov (predvolené: 3.0)
        
    Returns:
        pd.DataFrame: DataFrame s booleovskými hodnotami indikujúcimi outliery
    """
    print(f"[INFO] Detekcia outlierov pomocou Z-score (prah: {threshold})...")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Výpočet Z-score
    outlier_mask = pd.DataFrame(index=df.index)
    
    for col in columns:
        if col in df.columns:
            filled = df[col].fillna(df[col].mean())
            outlier_mask[col] = np.abs(stats.zscore(filled)) > threshold
    
    # Súhrn outlierov
    total_outliers = outlier_mask.sum().sum()
    print(f"[OK] Nájdených {total_outliers} odľahlých hodnôt v {len(columns)} stĺpcoch")
    
    for col in columns:
        if col in outlier_mask.columns:
            n_outliers = outlier_mask[col].sum()
            if n_outliers > 0:
                print(f"  - {col}: {n_outliers} outlierov ({n_outliers/len(df)*100:.2f}%)")
    
    return outlier_mask


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str] = None,
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Detekuje odľahlé hodnoty pomocou IQR (Interquartile Range) metódy.
    
    IQR = Q3 - Q1
    Outlier ak: hodnota < Q1 - multiplier*IQR alebo hodnota > Q3 + multiplier*IQR
    
    Args:
        df (pd.DataFrame): Vstupný DataFrame
        columns (List[str], optional): Stĺpce na analýzu
        multiplier (float): Násobiteľ IQR (predvolené: 1.5)
        
    Returns:
        pd.DataFrame: DataFrame s booleovskými hodnotami indikujúcimi outliery
    """
    print(f"[INFO] Detekcia outlierov pomocou IQR metódy (násobiteľ: {multiplier})...")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_mask = pd.DataFrame(index=df.index)
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    # Súhrn outlierov
    total_outliers = outlier_mask.sum().sum()
    print(f"[OK] Nájdených {total_outliers} odľahlých hodnôt v {len(columns)} stĺpcoch")
    
    return outlier_mask


def handle_outliers(
    df: pd.DataFrame,
    outlier_mask: pd.DataFrame,
    method: str = "clip"
) -> pd.DataFrame:
    """
    Ošetrí odľahlé hodnoty podľa zvolenej metódy.
    
    Args:
        df (pd.DataFrame): Vstupný DataFrame
        outlier_mask (pd.DataFrame): Maska outlierov
        method (str): Metóda ošetrenia - "clip", "remove", "median", "mean"
        
    Returns:
        pd.DataFrame: DataFrame s ošetrenými outliermi
    """
    print(f"[INFO] Ošetrenie outlierov metódou '{method}'...")
    
    df_handled = df.copy()
    
    for col in outlier_mask.columns:
        if col not in df_handled.columns:
            continue
            
        mask = outlier_mask[col]
        n_outliers = mask.sum()
        
        if n_outliers == 0:
            continue
        
        if method == "clip":
            # Orezanie na percentily
            lower = df_handled[col].quantile(0.01)
            upper = df_handled[col].quantile(0.99)
            df_handled[col] = df_handled[col].clip(lower=lower, upper=upper)
            
        elif method == "remove":
            # Odstránenie riadkov s outliermi
            df_handled = df_handled[~mask]
            
        elif method == "median":
            # Nahradenie mediánom
            median_val = df_handled[col].median()
            df_handled.loc[mask, col] = median_val
            
        elif method == "mean":
            # Nahradenie priemerom
            mean_val = df_handled[col].mean()
            df_handled.loc[mask, col] = mean_val
        
        else:
            print(f"[VAROVANIE] Neznáma metóda '{method}'. Outliery ponechané.")
    
    print(f"[OK] Outliers ošetrené. Výsledný počet riadkov: {len(df_handled)}")
    
    return df_handled


def calculate_returns(
    df: pd.DataFrame,
    price_column: str = "Close",
    periods: List[int] = None
) -> pd.DataFrame:
    """
    Vypočíta výnosy (returns) pre rôzne časové periódy.
    
    Args:
        df (pd.DataFrame): DataFrame s cenovými dátami
        price_column (str): Názov stĺpca s cenami
        periods (List[int], optional): Zoznam periód pre výpočet výnosov
        
    Returns:
        pd.DataFrame: DataFrame s pridanými stĺpcami výnosov
    """
    print("[INFO] Výpočet výnosov...")
    
    if periods is None:
        periods = [1, 5, 10, 20]  # Denné, týždenné, 2-týždenné, mesačné
    
    df_returns = df.copy()
    
    for period in periods:
        # Percentuálna zmena (jednoduchý výnos)
        col_name = f"Return_{period}d"
        df_returns[col_name] = df_returns[price_column].pct_change(periods=period) * 100
        
        # Logaritmický výnos
        log_col_name = f"LogReturn_{period}d"
        df_returns[log_col_name] = np.log(
            df_returns[price_column] / df_returns[price_column].shift(period)
        ) * 100
        
        print(f"  - Vypočítaný výnos pre {period}-dňovú periódu")
    
    print("[OK] Výnosy vypočítané")
    
    return df_returns


def calculate_volatility(
    df: pd.DataFrame,
    return_column: str = "Return_1d",
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Vypočíta historickú volatilitu pre rôzne okná.
    
    Volatilita je meraná ako štandardná odchýlka výnosov.
    
    Args:
        df (pd.DataFrame): DataFrame s výnosmi
        return_column (str): Stĺpec s výnosmi
        windows (List[int], optional): Zoznam okien pre výpočet volatility
        
    Returns:
        pd.DataFrame: DataFrame s pridanými stĺpcami volatility
    """
    print("[INFO] Výpočet volatility...")
    
    if windows is None:
        windows = [5, 10, 20, 60]  # Týždenná, 2-týždenná, mesačná, štvrťročná
    
    df_vol = df.copy()
    
    for window in windows:
        col_name = f"Volatility_{window}d"
        df_vol[col_name] = df_vol[return_column].rolling(window=window).std()
        
        # Annualizovaná volatilita (predpoklad 252 obchodných dní)
        ann_col_name = f"AnnVolatility_{window}d"
        df_vol[ann_col_name] = df_vol[col_name] * np.sqrt(252)
        
        print(f"  - Vypočítaná volatilita pre {window}-dňové okno")
    
    print("[OK] Volatilita vypočítaná")
    
    return df_vol


def create_target_variables(
    df: pd.DataFrame,
    price_column: str = "Close",
    horizons: List[int] = None,
    direction_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Vytvorí cieľové premenné pre predikčné modely.
    
    Vytvára:
    - Regresné ciele (budúca cena, budúci výnos)
    - Klasifikačné ciele (smer pohybu: hore/dole)
    
    Args:
        df (pd.DataFrame): DataFrame s dátami
        price_column (str): Stĺpec s cenami
        horizons (List[int], optional): Horizonty predikcie v dňoch
        direction_threshold (float): Prah pre klasifikáciu smeru
        
    Returns:
        pd.DataFrame: DataFrame s pridanými cieľovými premennými
    """
    print("[INFO] Vytváranie cieľových premenných...")
    
    if horizons is None:
        horizons = config.PREDICTION_HORIZONS
    
    df_target = df.copy()
    
    for horizon in horizons:
        # Budúca cena (pre regresiu)
        future_price_col = f"Target_Price_{horizon}d"
        df_target[future_price_col] = df_target[price_column].shift(-horizon)
        
        # Budúci výnos (pre regresiu)
        future_return_col = f"Target_Return_{horizon}d"
        df_target[future_return_col] = (
            (df_target[future_price_col] - df_target[price_column]) / 
            df_target[price_column] * 100
        )
        
        # Smer pohybu (pre klasifikáciu)
        # 1 = hore, 0 = dole
        direction_col = f"Target_Direction_{horizon}d"
        df_target[direction_col] = (
            df_target[future_return_col] > direction_threshold
        ).astype(int)
        
        print(f"  - Vytvorené ciele pre {horizon}-dňový horizont")
        
        # Štatistika smeru
        if direction_col in df_target.columns:
            up_pct = df_target[direction_col].mean() * 100
            print(f"    -> Pomer hore/dole: {up_pct:.1f}% / {100-up_pct:.1f}%")
    
    print("[OK] Cieľové premenné vytvorené")
    
    return df_target


def normalize_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    scaler_type: str = "standard",
    fit: bool = True,
    scaler: object = None
) -> Tuple[pd.DataFrame, object]:
    """
    Normalizuje features pomocou zvoleného scalera.
    
    Args:
        df (pd.DataFrame): DataFrame s features
        feature_columns (List[str]): Zoznam stĺpcov na normalizáciu
        scaler_type (str): Typ scalera - "standard", "minmax", "robust"
        fit (bool): Ak True, scaler sa natrénuje na dátach
        scaler (object, optional): Predtrénovaný scaler
        
    Returns:
        Tuple[pd.DataFrame, object]: Normalizovaný DataFrame a scaler
    """
    print(f"[INFO] Normalizácia features pomocou {scaler_type} scalera...")
    
    df_norm = df.copy()
    
    # Výber stĺpcov ktoré existujú
    valid_columns = [col for col in feature_columns if col in df_norm.columns]
    
    if not valid_columns:
        print("[VAROVANIE] Žiadne platné stĺpce na normalizáciu")
        return df_norm, scaler
    
    # Vytvorenie alebo použitie existujúceho scalera
    if scaler is None:
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            print(f"[VAROVANIE] Neznámy typ scalera. Použije sa StandardScaler.")
            scaler = StandardScaler()
    
    # Transformácia dát
    if fit:
        df_norm[valid_columns] = scaler.fit_transform(df_norm[valid_columns])
        print(f"  - Scaler natrénovaný a aplikovaný na {len(valid_columns)} stĺpcov")
    else:
        df_norm[valid_columns] = scaler.transform(df_norm[valid_columns])
        print(f"  - Scaler aplikovaný na {len(valid_columns)} stĺpcov")
    
    print("[OK] Normalizácia dokončená")
    
    return df_norm, scaler


def split_data(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str] = None,
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None,
    shuffle: bool = False,
    random_state: int = None,
    scale_features: bool = True
) -> Dict[str, np.ndarray]:
    """
    Rozdelí dáta na trénovaciu, validačnú a testovaciu množinu.
    
    Pre časové rady je odporúčané nepoužívať shuffle, aby sa zachovala
    časová postupnosť dát.
    
    Args:
        df (pd.DataFrame): DataFrame s dátami
        target_column (str): Názov cieľového stĺpca
        feature_columns (List[str], optional): Zoznam feature stĺpcov
        train_ratio (float, optional): Pomer trénovacej množiny
        val_ratio (float, optional): Pomer validačnej množiny
        test_ratio (float, optional): Pomer testovacej množiny
        shuffle (bool): Ak True, dáta sa zamiešajú
        random_state (int, optional): Seed pre reprodukovateľnosť
        scale_features (bool): Ak True, aplikuje StandardScaler fitovaný
            výlučne na trénovacích dátach (zabraňuje data leakage)
        
    Returns:
        Dict: Slovník s rozdelenými dátami (X_train, X_val, X_test,
              y_train, y_val, y_test, feature_names, scaler)
    """
    print("[INFO] Rozdelenie dát na trénovaciu, validačnú a testovaciu množinu...")
    
    # Predvolené hodnoty z konfigurácie
    if train_ratio is None:
        train_ratio = config.TRAIN_RATIO
    if val_ratio is None:
        val_ratio = config.VAL_RATIO
    if test_ratio is None:
        test_ratio = config.TEST_RATIO
    if random_state is None:
        random_state = config.RANDOM_SEED
    
    # Kontrola súčtu
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"[VAROVANIE] Súčet pomerov ({total_ratio}) nie je 1.0. Normalizujem.")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # Odstránenie riadkov s NaN v cieľovom stĺpci
    df_clean = df.dropna(subset=[target_column])
    
    # Ak nie sú špecifikované feature stĺpce, použijú sa všetky numerické okrem cieľového
    if feature_columns is None:
        feature_columns = [
            col for col in df_clean.select_dtypes(include=[np.number]).columns
            if col != target_column and not col.startswith('Target_')
        ]
    
    # Výber platných stĺpcov
    valid_features = [col for col in feature_columns if col in df_clean.columns]
    
    # Odstránenie riadkov s NaN vo features
    df_clean = df_clean.dropna(subset=valid_features)
    
    # Príprava X a y
    X = df_clean[valid_features].values
    y = df_clean[target_column].values
    
    n_samples = len(X)
    print(f"  - Celkový počet vzoriek: {n_samples}")
    print(f"  - Počet features: {len(valid_features)}")
    
    if shuffle:
        # Rozdelenie s miešaním
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(val_ratio + test_ratio),
            random_state=random_state
        )
        
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=relative_test_ratio,
            random_state=random_state
        )
    else:
        # Rozdelenie bez miešania (chronologické)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
    
    # Škálovanie features (fit len na train, transform na val/test)
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        print(f"  - Features škálované pomocou StandardScaler (fit na train)")

    # Výsledky
    result = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': valid_features,
        'scaler': scaler
    }
    
    print(f"  - Trénovacia množina: {len(X_train)} vzoriek ({len(X_train)/n_samples*100:.1f}%)")
    print(f"  - Validačná množina: {len(X_val)} vzoriek ({len(X_val)/n_samples*100:.1f}%)")
    print(f"  - Testovacia množina: {len(X_test)} vzoriek ({len(X_test)/n_samples*100:.1f}%)")
    print("[OK] Rozdelenie dát dokončené")
    
    return result


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vytvorí sekvencie pre rekurentné neurónové siete (LSTM, GRU).
    
    Args:
        X (np.ndarray): Feature matica
        y (np.ndarray): Cieľový vektor
        sequence_length (int): Dĺžka sekvencie
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Sekvencie X a y
    """
    print(f"[INFO] Vytváranie sekvencií s dĺžkou {sequence_length}...")
    
    X_seq = []
    y_seq = []
    
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"  - Vytvorených {len(X_seq)} sekvencií")
    print(f"  - Tvar X: {X_seq.shape}")
    print(f"  - Tvar y: {y_seq.shape}")
    print("[OK] Sekvencie vytvorené")
    
    return X_seq, y_seq


def preprocess_pipeline(
    df: pd.DataFrame,
    ticker: str = None,
    save_to_file: bool = True
) -> pd.DataFrame:
    """
    Kompletný pipeline pre predspracovanie dát.
    
    Vykoná všetky kroky predspracovania:
    1. Čistenie dát
    2. Detekcia a ošetrenie outlierov
    3. Výpočet výnosov a volatility
    4. Vytvorenie cieľových premenných
    
    Args:
        df (pd.DataFrame): Surové dáta
        ticker (str, optional): Symbol akcie
        save_to_file (bool): Ak True, uloží spracované dáta
        
    Returns:
        pd.DataFrame: Spracované dáta
    """
    print("\n" + "=" * 60)
    print(f"PREDSPRACOVANIE DÁT" + (f" PRE {ticker}" if ticker else ""))
    print("=" * 60)
    
    # 1. Čistenie dát
    df_processed = clean_data(df)
    
    # 2. Detekcia outlierov
    price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    existing_price_cols = [col for col in price_columns if col in df_processed.columns]
    
    outlier_mask = detect_outliers_iqr(df_processed, columns=existing_price_cols)
    df_processed = handle_outliers(df_processed, outlier_mask, method="clip")
    
    # 3. Výpočet výnosov
    df_processed = calculate_returns(df_processed, price_column="Close")
    
    # 4. Výpočet volatility
    if 'Return_1d' in df_processed.columns:
        df_processed = calculate_volatility(df_processed, return_column="Return_1d")
    
    # 5. Vytvorenie cieľových premenných
    df_processed = create_target_variables(df_processed, price_column="Close")
    
    # 6. Odstránenie riadkov s NaN (vzniknutých pri výpočtoch)
    initial_len = len(df_processed)
    df_processed = df_processed.dropna()
    print(f"[INFO] Odstránených {initial_len - len(df_processed)} riadkov s NaN hodnotami")
    
    # Uloženie
    if save_to_file and ticker:
        filepath = os.path.join(config.DATA_DIR, f"{ticker}_preprocessed.csv")
        df_processed.to_csv(filepath, index=False)
        print(f"[OK] Predspracované dáta uložené do {filepath}")
    
    print("\n" + "=" * 60)
    print("PREDSPRACOVANIE DOKONČENÉ")
    print(f"Výsledný počet záznamov: {len(df_processed)}")
    print("=" * 60)
    
    return df_processed


def get_preprocessing_summary(df: pd.DataFrame) -> Dict:
    """
    Vytvorí súhrn predspracovaných dát.
    
    Args:
        df (pd.DataFrame): Predspracované dáta
        
    Returns:
        Dict: Súhrn štatistík
    """
    summary = {
        'pocet_zaznamov': len(df),
        'pocet_stlpcov': len(df.columns),
        'casovy_rozsah': None,
        'chybajuce_hodnoty': df.isnull().sum().sum(),
        'statistiky': {}
    }
    
    if 'Datum' in df.columns:
        summary['casovy_rozsah'] = f"{df['Datum'].min()} - {df['Datum'].max()}"
    
    # Základné štatistiky pre kľúčové stĺpce
    key_columns = ['Close', 'Return_1d', 'Volatility_20d']
    for col in key_columns:
        if col in df.columns:
            summary['statistiky'][col] = {
                'priemer': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
    
    return summary


# =============================================================================
# HLAVNÁ FUNKCIA PRE TESTOVANIE MODULU
# =============================================================================

if __name__ == "__main__":
    import data_downloader
    
    print("=" * 60)
    print("TESTOVANIE MODULU DATA_PREPROCESSING")
    print("=" * 60)
    
    # Načítanie testovacích dát
    try:
        df = data_downloader.load_stock_data("AAPL")
    except FileNotFoundError:
        print("[INFO] Dáta nie sú k dispozícii, sťahujem...")
        df = data_downloader.download_stock_data("AAPL")
    
    # Test predspracovania
    df_processed = preprocess_pipeline(df, ticker="AAPL")
    
    # Súhrn
    summary = get_preprocessing_summary(df_processed)
    print("\nSúhrn predspracovaných dát:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
