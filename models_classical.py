# -*- coding: utf-8 -*-
"""
Modul pre klasické algoritmy strojového učenia.

Tento modul implementuje nasledujúce modely:
- Random Forest (klasifikácia aj regresia)
- XGBoost (klasifikácia aj regresia)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Gradient Boosting
- Decision Tree
- AdaBoost

Autor: Dominika Melicherová
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

# Sklearn imports
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

# XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[VAROVANIE] XGBoost nie je nainštalovaný. XGBoost modely nebudú dostupné.")

# Import konfigurácie
import config


# =============================================================================
# ABSTRAKTNÁ ZÁKLADNÁ TRIEDA PRE MODELY
# =============================================================================

class BaseModel(ABC):
    """
    Abstraktná základná trieda pre všetky modely.
    
    Definuje spoločné rozhranie pre trénovanie, predikciu a vyhodnotenie.
    """
    
    def __init__(self, name: str, model_type: str = "classification"):
        """
        Inicializácia základnej triedy.
        
        Args:
            name (str): Názov modelu
            model_type (str): Typ modelu - "classification" alebo "regression"
        """
        self.name = name
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.scaler = None
        self.feature_names = None
        
    @abstractmethod
    def build_model(self, **params):
        """Vytvorí model s danými parametrami."""
        pass
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str] = None
    ) -> None:
        """
        Natrénuje model na trénovacích dátach.
        
        Args:
            X_train (np.ndarray): Trénovacie features
            y_train (np.ndarray): Trénovacie labels
            feature_names (List[str], optional): Názvy features
        """
        print(f"[INFO] Trénovanie modelu {self.name}...")
        
        if self.model is None:
            raise ValueError("Model nie je vytvorený. Najprv zavolajte build_model().")
        
        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print(f"[OK] Model {self.name} úspešne natrénovaný")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Vykoná predikciu na dátach.
        
        Args:
            X (np.ndarray): Vstupné dáta
            
        Returns:
            np.ndarray: Predikcie
        """
        if not self.is_trained:
            raise ValueError("Model nie je natrénovaný. Najprv zavolajte train().")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Vráti pravdepodobnosti tried (len pre klasifikáciu).
        
        Args:
            X (np.ndarray): Vstupné dáta
            
        Returns:
            np.ndarray: Pravdepodobnosti tried
        """
        if self.model_type != "classification":
            raise ValueError("predict_proba je dostupné len pre klasifikačné modely.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError(f"Model {self.name} nepodporuje predict_proba.")
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Vráti dôležitosť features (ak je dostupná).
        
        Returns:
            np.ndarray alebo None: Dôležitosť features
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).flatten()
        return None
    
    def save_model(self, filepath: str = None) -> str:
        """
        Uloží model do súboru.
        
        Args:
            filepath (str, optional): Cesta k súboru
            
        Returns:
            str: Cesta k uloženému súboru
        """
        if filepath is None:
            filepath = os.path.join(config.MODELS_DIR, f"{self.name.lower().replace(' ', '_')}.joblib")
        
        model_data = {
            'model': self.model,
            'name': self.name,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, filepath)
        print(f"[OK] Model uložený do {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str) -> None:
        """
        Načíta model zo súboru.
        
        Args:
            filepath (str): Cesta k súboru
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.name = model_data['name']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        self.scaler = model_data.get('scaler', None)
        
        print(f"[OK] Model načítaný z {filepath}")


# =============================================================================
# RANDOM FOREST
# =============================================================================

class RandomForestModel(BaseModel):
    """
    Random Forest model pre klasifikáciu alebo regresiu.
    
    Random Forest je ensemble metóda, ktorá kombinuje viacero rozhodovacích
    stromov pre robustnejšie predikcie.
    """
    
    def __init__(self, model_type: str = "classification"):
        """
        Inicializácia Random Forest modelu.
        
        Args:
            model_type (str): "classification" alebo "regression"
        """
        name = f"Random Forest ({'Klasifikácia' if model_type == 'classification' else 'Regresia'})"
        super().__init__(name, model_type)
        
    def build_model(self, **params) -> None:
        """
        Vytvorí Random Forest model.
        
        Args:
            **params: Parametre modelu (n_estimators, max_depth, atď.)
        """
        # Použitie predvolených parametrov z konfigurácie
        default_params = config.RF_PARAMS.copy()
        default_params.update(params)
        
        if self.model_type == "classification":
            self.model = RandomForestClassifier(class_weight='balanced', **default_params)
        else:
            self.model = RandomForestRegressor(**default_params)
        
        print(f"[OK] {self.name} model vytvorený s parametrami: {default_params}")


# =============================================================================
# XGBOOST
# =============================================================================

class XGBoostModel(BaseModel):
    """
    XGBoost model pre klasifikáciu alebo regresiu.
    
    XGBoost je vysoko efektívna implementácia gradient boosting,
    známa pre svoju rýchlosť a výkon.
    """
    
    def __init__(self, model_type: str = "classification"):
        """
        Inicializácia XGBoost modelu.
        
        Args:
            model_type (str): "classification" alebo "regression"
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost nie je nainštalovaný. Nainštalujte ho pomocou: pip install xgboost")
        
        name = f"XGBoost ({'Klasifikácia' if model_type == 'classification' else 'Regresia'})"
        super().__init__(name, model_type)
        
    def build_model(self, **params) -> None:
        """
        Vytvorí XGBoost model.
        
        Args:
            **params: Parametre modelu
        """
        default_params = config.XGB_PARAMS.copy()
        default_params.update(params)
        
        if self.model_type == "classification":
            self.model = XGBClassifier(**default_params, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1)
        else:
            self.model = XGBRegressor(**default_params)
        
        print(f"[OK] {self.name} model vytvorený s parametrami: {default_params}")


# =============================================================================
# SUPPORT VECTOR MACHINE
# =============================================================================

class SVMModel(BaseModel):
    """
    Support Vector Machine model pre klasifikáciu alebo regresiu.
    
    SVM hľadá optimálnu hranicu (hyperrovinu) medzi triedami.
    """
    
    def __init__(self, model_type: str = "classification"):
        """
        Inicializácia SVM modelu.
        
        Args:
            model_type (str): "classification" alebo "regression"
        """
        name = f"SVM ({'Klasifikácia' if model_type == 'classification' else 'Regresia'})"
        super().__init__(name, model_type)
        
    def build_model(self, **params) -> None:
        """
        Vytvorí SVM model.
        
        Args:
            **params: Parametre modelu
        """
        default_params = config.SVM_PARAMS.copy()
        default_params.update(params)
        
        if self.model_type == "classification":
            # Pridanie probability=True pre predict_proba
            default_params['probability'] = True
            self.model = SVC(class_weight='balanced', **default_params)
        else:
            # SVR nepodporuje random_state ani probability
            default_params.pop('random_state', None)
            default_params.pop('probability', None)
            self.model = SVR(**default_params)
        
        print(f"[OK] {self.name} model vytvorený s parametrami: {default_params}")


# =============================================================================
# K-NEAREST NEIGHBORS
# =============================================================================

class KNNModel(BaseModel):
    """
    K-Nearest Neighbors model pre klasifikáciu alebo regresiu.
    
    KNN klasifikuje vzorky podľa majority najbližších susedov.
    """
    
    def __init__(self, model_type: str = "classification"):
        """
        Inicializácia KNN modelu.
        
        Args:
            model_type (str): "classification" alebo "regression"
        """
        name = f"KNN ({'Klasifikácia' if model_type == 'classification' else 'Regresia'})"
        super().__init__(name, model_type)
        
    def build_model(self, **params) -> None:
        """
        Vytvorí KNN model.
        
        Args:
            **params: Parametre modelu
        """
        default_params = config.KNN_PARAMS.copy()
        default_params.update(params)
        
        if self.model_type == "classification":
            self.model = KNeighborsClassifier(**default_params)
        else:
            self.model = KNeighborsRegressor(**default_params)
        
        print(f"[OK] {self.name} model vytvorený s parametrami: {default_params}")


# =============================================================================
# LOGISTIC REGRESSION / LINEAR REGRESSION
# =============================================================================

class LogisticRegressionModel(BaseModel):
    """
    Logistická regresia pre klasifikáciu.
    
    Jednoduchý ale efektívny lineárny model pre binárnu klasifikáciu.
    """
    
    def __init__(self):
        """Inicializácia Logistickej regresie."""
        super().__init__("Logistická regresia", "classification")
        
    def build_model(self, **params) -> None:
        """
        Vytvorí model logistickej regresie.
        
        Args:
            **params: Parametre modelu
        """
        default_params = config.LOGREG_PARAMS.copy()
        default_params.update(params)
        
        self.model = LogisticRegression(class_weight='balanced', **default_params)
        
        print(f"[OK] {self.name} model vytvorený s parametrami: {default_params}")


class RidgeRegressionModel(BaseModel):
    """
    Ridge regresia (L2 regularizácia) pre regresiu.
    """
    
    def __init__(self):
        """Inicializácia Ridge regresie."""
        super().__init__("Ridge regresia", "regression")
        
    def build_model(self, alpha: float = 1.0, **params) -> None:
        """
        Vytvorí model Ridge regresie.
        
        Args:
            alpha (float): Sila regularizácie
            **params: Ďalšie parametre
        """
        self.model = Ridge(alpha=alpha, random_state=config.RANDOM_SEED, **params)
        print(f"[OK] {self.name} model vytvorený s alpha={alpha}")


class LassoRegressionModel(BaseModel):
    """
    Lasso regresia (L1 regularizácia) pre regresiu.
    """
    
    def __init__(self):
        """Inicializácia Lasso regresie."""
        super().__init__("Lasso regresia", "regression")
        
    def build_model(self, alpha: float = 1.0, **params) -> None:
        """
        Vytvorí model Lasso regresie.
        
        Args:
            alpha (float): Sila regularizácie
            **params: Ďalšie parametre
        """
        self.model = Lasso(alpha=alpha, random_state=config.RANDOM_SEED, **params)
        print(f"[OK] {self.name} model vytvorený s alpha={alpha}")


# =============================================================================
# GRADIENT BOOSTING
# =============================================================================

class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting model pre klasifikáciu alebo regresiu.
    
    Gradient Boosting sekvenčne pridáva slabé modely,
    pričom každý opravuje chyby predchádzajúcich.
    """
    
    def __init__(self, model_type: str = "classification"):
        """
        Inicializácia Gradient Boosting modelu.
        
        Args:
            model_type (str): "classification" alebo "regression"
        """
        name = f"Gradient Boosting ({'Klasifikácia' if model_type == 'classification' else 'Regresia'})"
        super().__init__(name, model_type)
        
    def build_model(self, **params) -> None:
        """
        Vytvorí Gradient Boosting model.
        
        Args:
            **params: Parametre modelu
        """
        default_params = config.GB_PARAMS.copy()
        default_params.update(params)
        
        if self.model_type == "classification":
            self.model = GradientBoostingClassifier(**default_params)
        else:
            self.model = GradientBoostingRegressor(**default_params)
        
        print(f"[OK] {self.name} model vytvorený s parametrami: {default_params}")


# =============================================================================
# DECISION TREE
# =============================================================================

class DecisionTreeModel(BaseModel):
    """
    Rozhodovací strom pre klasifikáciu alebo regresiu.
    
    Jednoduchý interpretovateľný model založený na postupnom delení dát.
    """
    
    def __init__(self, model_type: str = "classification"):
        """
        Inicializácia Rozhodovacieho stromu.
        
        Args:
            model_type (str): "classification" alebo "regression"
        """
        name = f"Rozhodovací strom ({'Klasifikácia' if model_type == 'classification' else 'Regresia'})"
        super().__init__(name, model_type)
        
    def build_model(self, max_depth: int = 10, **params) -> None:
        """
        Vytvorí model rozhodovacieho stromu.
        
        Args:
            max_depth (int): Maximálna hĺbka stromu
            **params: Ďalšie parametre
        """
        if self.model_type == "classification":
            self.model = DecisionTreeClassifier(
                class_weight='balanced',
                max_depth=max_depth,
                random_state=config.RANDOM_SEED,
                **params
            )
        else:
            self.model = DecisionTreeRegressor(
                max_depth=max_depth,
                random_state=config.RANDOM_SEED,
                **params
            )
        
        print(f"[OK] {self.name} model vytvorený s max_depth={max_depth}")


# =============================================================================
# ADABOOST
# =============================================================================

class AdaBoostModel(BaseModel):
    """
    AdaBoost model pre klasifikáciu alebo regresiu.
    
    AdaBoost adaptívne zvyšuje váhu zle klasifikovaných vzoriek.
    """
    
    def __init__(self, model_type: str = "classification"):
        """
        Inicializácia AdaBoost modelu.
        
        Args:
            model_type (str): "classification" alebo "regression"
        """
        name = f"AdaBoost ({'Klasifikácia' if model_type == 'classification' else 'Regresia'})"
        super().__init__(name, model_type)
        
    def build_model(self, n_estimators: int = 50, learning_rate: float = 1.0, **params) -> None:
        """
        Vytvorí AdaBoost model.
        
        Args:
            n_estimators (int): Počet estimátorov
            learning_rate (float): Rýchlosť učenia
            **params: Ďalšie parametre
        """
        if self.model_type == "classification":
            self.model = AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=config.RANDOM_SEED,
                **params
            )
        else:
            self.model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=config.RANDOM_SEED,
                **params
            )
        
        print(f"[OK] {self.name} model vytvorený s n_estimators={n_estimators}")


# =============================================================================
# POMOCNÉ FUNKCIE
# =============================================================================

def create_all_classification_models() -> Dict[str, BaseModel]:
    """
    Vytvorí všetky dostupné klasifikačné modely.
    
    Returns:
        Dict[str, BaseModel]: Slovník modelov
    """
    print("\n" + "=" * 60)
    print("VYTVÁRANIE KLASIFIKAČNÝCH MODELOV")
    print("=" * 60)
    
    models = {}
    
    # Random Forest
    rf = RandomForestModel("classification")
    rf.build_model()
    models['Random Forest'] = rf
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBoostModel("classification")
        xgb.build_model()
        models['XGBoost'] = xgb
    
    # SVM
    svm = SVMModel("classification")
    svm.build_model()
    models['SVM'] = svm
    
    # KNN
    knn = KNNModel("classification")
    knn.build_model()
    models['KNN'] = knn
    
    # Logistic Regression
    logreg = LogisticRegressionModel()
    logreg.build_model()
    models['Logistická regresia'] = logreg
    
    # Gradient Boosting
    gb = GradientBoostingModel("classification")
    gb.build_model()
    models['Gradient Boosting'] = gb
    
    # Decision Tree
    dt = DecisionTreeModel("classification")
    dt.build_model()
    models['Rozhodovací strom'] = dt
    
    # AdaBoost
    ada = AdaBoostModel("classification")
    ada.build_model()
    models['AdaBoost'] = ada
    
    print(f"\n[OK] Vytvorených {len(models)} klasifikačných modelov")
    
    return models


def create_all_regression_models() -> Dict[str, BaseModel]:
    """
    Vytvorí všetky dostupné regresné modely.
    
    Returns:
        Dict[str, BaseModel]: Slovník modelov
    """
    print("\n" + "=" * 60)
    print("VYTVÁRANIE REGRESNÝCH MODELOV")
    print("=" * 60)
    
    models = {}
    
    # Random Forest
    rf = RandomForestModel("regression")
    rf.build_model()
    models['Random Forest'] = rf
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBoostModel("regression")
        xgb.build_model()
        models['XGBoost'] = xgb
    
    # SVM
    svm = SVMModel("regression")
    svm.build_model()
    models['SVM'] = svm
    
    # KNN
    knn = KNNModel("regression")
    knn.build_model()
    models['KNN'] = knn
    
    # Ridge
    ridge = RidgeRegressionModel()
    ridge.build_model()
    models['Ridge'] = ridge
    
    # Lasso
    lasso = LassoRegressionModel()
    lasso.build_model()
    models['Lasso'] = lasso
    
    # Gradient Boosting
    gb = GradientBoostingModel("regression")
    gb.build_model()
    models['Gradient Boosting'] = gb
    
    # Decision Tree
    dt = DecisionTreeModel("regression")
    dt.build_model()
    models['Rozhodovací strom'] = dt
    
    # AdaBoost
    ada = AdaBoostModel("regression")
    ada.build_model()
    models['AdaBoost'] = ada
    
    print(f"\n[OK] Vytvorených {len(models)} regresných modelov")
    
    return models


def train_all_models(
    models: Dict[str, BaseModel],
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str] = None
) -> Dict[str, BaseModel]:
    """
    Natrénuje všetky modely v slovníku.
    
    Args:
        models (Dict[str, BaseModel]): Slovník modelov
        X_train (np.ndarray): Trénovacie features
        y_train (np.ndarray): Trénovacie labels
        feature_names (List[str], optional): Názvy features
        
    Returns:
        Dict[str, BaseModel]: Slovník natrénovaných modelov
    """
    print("\n" + "=" * 60)
    print("TRÉNOVANIE MODELOV")
    print("=" * 60)
    
    trained_models = {}
    
    for name, model in models.items():
        try:
            model.train(X_train, y_train, feature_names)
            trained_models[name] = model
        except Exception as e:
            print(f"[CHYBA] Nepodarilo sa natrénovať {name}: {str(e)}")
    
    print(f"\n[OK] Úspešne natrénovaných {len(trained_models)}/{len(models)} modelov")
    
    return trained_models


def cross_validate_model(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = None
) -> Dict[str, float]:
    """
    Vykoná krížovú validáciu modelu.
    
    Args:
        model (BaseModel): Model na validáciu
        X (np.ndarray): Features
        y (np.ndarray): Labels
        cv (int): Počet foldov
        scoring (str, optional): Metrika
        
    Returns:
        Dict[str, float]: Výsledky krížovej validácie
    """
    print(f"[INFO] Krížová validácia pre {model.name} ({cv} foldov)...")
    
    if scoring is None:
        scoring = 'accuracy' if model.model_type == 'classification' else 'neg_mean_squared_error'
    
    scores = cross_val_score(model.model, X, y, cv=cv, scoring=scoring)
    
    results = {
        'priemer': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'vsetky_skore': scores.tolist()
    }
    
    print(f"[OK] {model.name}: {scoring} = {results['priemer']:.4f} (+/- {results['std']:.4f})")
    
    return results


def hyperparameter_tuning(
    model: BaseModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List],
    cv: int = 5,
    scoring: str = None
) -> Tuple[Dict, float]:
    """
    Vykoná ladenie hyperparametrov pomocou Grid Search.
    
    Args:
        model (BaseModel): Model na ladenie
        X_train (np.ndarray): Trénovacie features
        y_train (np.ndarray): Trénovacie labels
        param_grid (Dict[str, List]): Mriežka parametrov
        cv (int): Počet foldov
        scoring (str, optional): Metrika
        
    Returns:
        Tuple[Dict, float]: Najlepšie parametre a skóre
    """
    print(f"[INFO] Ladenie hyperparametrov pre {model.name}...")
    
    if scoring is None:
        scoring = 'accuracy' if model.model_type == 'classification' else 'neg_mean_squared_error'
    
    grid_search = GridSearchCV(
        model.model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"[OK] Najlepšie parametre: {grid_search.best_params_}")
    print(f"[OK] Najlepšie skóre: {grid_search.best_score_:.4f}")
    
    # Aktualizácia modelu s najlepšími parametrami
    model.model = grid_search.best_estimator_
    model.is_trained = True
    
    return grid_search.best_params_, grid_search.best_score_


# =============================================================================
# HLAVNÁ FUNKCIA PRE TESTOVANIE MODULU
# =============================================================================

if __name__ == "__main__":
    import data_downloader
    import data_preprocessing
    import feature_engineering
    
    print("=" * 60)
    print("TESTOVANIE MODULU MODELS_CLASSICAL")
    print("=" * 60)
    
    # Načítanie a príprava dát
    try:
        df = data_downloader.load_stock_data("AAPL")
    except FileNotFoundError:
        df = data_downloader.download_stock_data("AAPL")
    
    df_processed = data_preprocessing.preprocess_pipeline(df, ticker="AAPL", save_to_file=False)
    df_features = feature_engineering.create_all_features(df_processed, ticker="AAPL", save_to_file=False)
    
    # Príprava dát pre modely
    feature_cols = feature_engineering.get_feature_list(df_features)
    target_col = 'Target_Direction_252d'
    
    split_data = data_preprocessing.split_data(
        df_features,
        target_column=target_col,
        feature_columns=feature_cols
    )
    
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    
    # Test vytvorenia a trénovania modelov
    print("\n--- Test klasifikačných modelov ---")
    models = create_all_classification_models()
    
    # Trénovanie modelov ako ukážka
    for rf in models:
        models[rf].train(X_train, y_train, feature_cols)
        
        # Predikcia
        y_pred = models[rf].predict(X_test)
        
        # Výpočet accuracy
        accuracy = (y_pred == y_test).mean()
        print(f"\n[OK] {rf} test accuracy: {accuracy:.4f}")
        
        # Feature importance
        importance = models[rf].get_feature_importance()
        if importance is not None:
            print(f"\nTop 5 najdôležitejších features:")
            indices = np.argsort(importance)[::-1][:5]
            for i in indices:
                print(f"  {feature_cols[i]}: {importance[i]:.4f}")
