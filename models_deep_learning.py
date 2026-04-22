# -*- coding: utf-8 -*-
"""
Modul pre modely hlbokého učenia (Deep Learning).

Tento modul implementuje nasledujúce architektúry:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Jednoduchá viacvrstvová neurónová sieť (MLP)
- Kombinácie CNN-LSTM

Autor: Dominika Melicherová
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Potlačenie TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout, BatchNormalization,
        Bidirectional, Conv1D, MaxPooling1D
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[VAROVANIE] TensorFlow nie je nainštalovaný. Deep learning modely nebudú dostupné.")

# Import konfigurácie
import config


# =============================================================================
# ABSTRAKTNÁ ZÁKLADNÁ TRIEDA PRE DEEP LEARNING MODELY
# =============================================================================

class BaseDeepModel:
    """
    Abstraktná základná trieda pre deep learning modely.
    
    Poskytuje spoločné rozhranie a funkcionality pre všetky
    neurónové siete v tomto module.
    """
    
    def __init__(self, name: str, model_type: str = "regression"):
        """
        Inicializácia základnej triedy.
        
        Args:
            name (str): Názov modelu
            model_type (str): Typ modelu - "classification" alebo "regression"
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow nie je nainštalovaný. Nainštalujte ho pomocou: pip install tensorflow")
        
        self.name = name
        self.model_type = model_type
        self.model = None
        self.history = None
        self.is_trained = False
        self.is_sequential = True
        self.input_shape = None
        
    def compile_model(
        self,
        learning_rate: float = None,
        loss: str = None,
        metrics: List[str] = None
    ) -> None:
        """
        Skompiluje model s optimizerom a loss funkciou.
        
        Args:
            learning_rate (float, optional): Rýchlosť učenia
            loss (str, optional): Loss funkcia
            metrics (List[str], optional): Metriky na sledovanie
        """
        if learning_rate is None:
            learning_rate = config.DL_LEARNING_RATE
        
        if loss is None:
            loss = 'binary_crossentropy' if self.model_type == 'classification' else 'mse'
        
        if metrics is None:
            metrics = ['accuracy'] if self.model_type == 'classification' else ['mae']
        
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        print(f"[OK] Model {self.name} skompilovaný s learning_rate={learning_rate}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = None,
        batch_size: int = None,
        early_stopping: bool = True,
        verbose: int = 1
    ) -> Dict:
        """
        Natrénuje model na dátach.
        
        Args:
            X_train (np.ndarray): Trénovacie features
            y_train (np.ndarray): Trénovacie labels
            X_val (np.ndarray, optional): Validačné features
            y_val (np.ndarray, optional): Validačné labels
            epochs (int, optional): Počet epoch
            batch_size (int, optional): Veľkosť batch
            early_stopping (bool): Použiť early stopping
            verbose (int): Úroveň výpisu
            
        Returns:
            Dict: História trénovania
        """
        print(f"[INFO] Trénovanie modelu {self.name}...")
        
        if epochs is None:
            epochs = config.DL_EPOCHS
        if batch_size is None:
            batch_size = config.DL_BATCH_SIZE
        
        # Callbacks
        callbacks = []
        
        if early_stopping:
            es = EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=config.DL_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(es)
        
        # Reduce learning rate on plateau
        lr_reducer = ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reducer)
        
        # Validačné dáta
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Trénovanie
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Konverzia histórie na slovník
        history_dict = {key: list(values) for key, values in self.history.history.items()}
        
        print(f"[OK] Model {self.name} úspešne natrénovaný za {len(self.history.history['loss'])} epoch")
        
        return history_dict
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Vykoná predikciu.
        
        Args:
            X (np.ndarray): Vstupné dáta
            
        Returns:
            np.ndarray: Predikcie
        """
        if not self.is_trained:
            raise ValueError("Model nie je natrénovaný.")
        
        predictions = self.model.predict(X, verbose=0)
        
        if self.model_type == 'classification':
            # Konverzia na triedy
            return (predictions > 0.5).astype(int).flatten()
        
        return predictions.flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Vráti pravdepodobnosti (len pre klasifikáciu).
        
        Args:
            X (np.ndarray): Vstupné dáta
            
        Returns:
            np.ndarray: Pravdepodobnosti
        """
        if self.model_type != 'classification':
            raise ValueError("predict_proba je len pre klasifikáciu.")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def save_model(self, filepath: str = None) -> str:
        """
        Uloží model do súboru.
        
        Args:
            filepath (str, optional): Cesta k súboru
            
        Returns:
            str: Cesta k uloženému modelu
        """
        if filepath is None:
            filepath = os.path.join(
                config.MODELS_DIR, 
                f"{self.name.lower().replace(' ', '_')}.keras"
            )
        
        self.model.save(filepath)
        print(f"[OK] Model uložený do {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str) -> None:
        """
        Načíta model zo súboru.
        
        Args:
            filepath (str): Cesta k súboru
        """
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        print(f"[OK] Model načítaný z {filepath}")
    
    def summary(self) -> None:
        """Vypíše súhrn architektúry modelu."""
        if self.model is not None:
            self.model.summary()


# =============================================================================
# LSTM MODEL
# =============================================================================

class LSTMModel(BaseDeepModel):
    """
    LSTM (Long Short-Term Memory) model pre časové rady.
    
    LSTM je typ rekurentnej neurónovej siete schopnej učiť sa
    dlhodobé závislosti v sekvenciách.
    """
    
    def __init__(self, model_type: str = "regression"):
        """
        Inicializácia LSTM modelu.
        
        Args:
            model_type (str): "classification" alebo "regression"
        """
        name = f"LSTM ({'Klasifikácia' if model_type == 'classification' else 'Regresia'})"
        super().__init__(name, model_type)
    
    def build_model(
        self,
        input_shape: Tuple[int, int],
        units: List[int] = None,
        dropout: float = None,
        bidirectional: bool = False
    ) -> None:
        """
        Vytvorí LSTM model.
        
        Args:
            input_shape (Tuple[int, int]): Tvar vstupu (sequence_length, n_features)
            units (List[int], optional): Počet jednotiek v každej LSTM vrstve
            dropout (float, optional): Dropout rate
            bidirectional (bool): Použiť obojsmernú LSTM
        """
        if units is None:
            units = config.LSTM_PARAMS['units']
        if dropout is None:
            dropout = config.LSTM_PARAMS['dropout']
        
        self.input_shape = input_shape
        
        model = Sequential()
        
        for i, n_units in enumerate(units):
            return_sequences = i < len(units) - 1  # True okrem poslednej vrstvy
            
            kwargs = {
                'units': n_units,
                'return_sequences': return_sequences,
                'kernel_regularizer': l2(0.001),
            }
            if i == 0:
                kwargs['input_shape'] = input_shape
            
            lstm_layer = LSTM(**kwargs)
            
            if bidirectional:
                lstm_layer = Bidirectional(lstm_layer)
            
            model.add(lstm_layer)
            model.add(Dropout(dropout))
            model.add(BatchNormalization())
        
        # Výstupná vrstva
        if self.model_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(1, activation='linear'))
        
        self.model = model
        
        print(f"[OK] {self.name} model vytvorený")
        print(f"  - Vstupný tvar: {input_shape}")
        print(f"  - LSTM vrstvy: {units}")
        print(f"  - Dropout: {dropout}")
        print(f"  - Obojsmerný: {bidirectional}")


class SimpleLSTMModel(BaseDeepModel):
    """
    Jednoduchší LSTM model - ľahšie laditeľný.
    """
    
    def __init__(self, model_type: str = "regression"):
        name = f"Simple LSTM ({'Klasifikácia' if model_type == 'classification' else 'Regresia'})"
        super().__init__(name, model_type)
    
    def build_model(
        self,
        input_shape: Tuple[int, int],
        lstm_units: int = 50,
        dense_units: int = 25,
        dropout: float = 0.2
    ) -> None:
        """
        Vytvorí jednoduchý LSTM model.
        
        Args:
            input_shape (Tuple[int, int]): Tvar vstupu
            lstm_units (int): Počet LSTM jednotiek
            dense_units (int): Počet jednotiek v Dense vrstve
            dropout (float): Dropout rate
        """
        self.input_shape = input_shape
        
        self.model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(lstm_units, return_sequences=False),
            Dropout(dropout),
            Dense(dense_units, activation='relu'),
            Dense(1, activation='sigmoid' if self.model_type == 'classification' else 'linear')
        ])
        
        print(f"[OK] {self.name} model vytvorený s {lstm_units} LSTM jednotkami")


# =============================================================================
# GRU MODEL
# =============================================================================

class GRUModel(BaseDeepModel):
    """
    GRU (Gated Recurrent Unit) model pre časové rady.
    
    GRU je podobná LSTM, ale s jednoduchšou architektúrou
    a menším počtom parametrov.
    """
    
    def __init__(self, model_type: str = "regression"):
        """
        Inicializácia GRU modelu.
        
        Args:
            model_type (str): "classification" alebo "regression"
        """
        name = f"GRU ({'Klasifikácia' if model_type == 'classification' else 'Regresia'})"
        super().__init__(name, model_type)
    
    def build_model(
        self,
        input_shape: Tuple[int, int],
        units: List[int] = None,
        dropout: float = None,
        bidirectional: bool = False
    ) -> None:
        """
        Vytvorí GRU model.
        
        Args:
            input_shape (Tuple[int, int]): Tvar vstupu
            units (List[int], optional): Počet jednotiek v každej GRU vrstve
            dropout (float, optional): Dropout rate
            bidirectional (bool): Použiť obojsmernú GRU
        """
        if units is None:
            units = config.GRU_PARAMS['units']
        if dropout is None:
            dropout = config.GRU_PARAMS['dropout']
        
        self.input_shape = input_shape
        
        model = Sequential()
        
        for i, n_units in enumerate(units):
            return_sequences = i < len(units) - 1
            
            kwargs = {
                'units': n_units,
                'return_sequences': return_sequences,
                'kernel_regularizer': l2(0.001),
            }
            if i == 0:
                kwargs['input_shape'] = input_shape
            
            gru_layer = GRU(**kwargs)
            
            if bidirectional:
                gru_layer = Bidirectional(gru_layer)
            
            model.add(gru_layer)
            model.add(Dropout(dropout))
            model.add(BatchNormalization())
        
        # Výstupná vrstva
        if self.model_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(1, activation='linear'))
        
        self.model = model
        
        print(f"[OK] {self.name} model vytvorený")
        print(f"  - Vstupný tvar: {input_shape}")
        print(f"  - GRU vrstvy: {units}")


class SimpleGRUModel(BaseDeepModel):
    """
    Jednoduchší GRU model.
    """
    
    def __init__(self, model_type: str = "regression"):
        name = f"Simple GRU ({'Klasifikácia' if model_type == 'classification' else 'Regresia'})"
        super().__init__(name, model_type)
    
    def build_model(
        self,
        input_shape: Tuple[int, int],
        gru_units: int = 50,
        dense_units: int = 25,
        dropout: float = 0.2
    ) -> None:
        """
        Vytvorí jednoduchý GRU model.
        """
        self.input_shape = input_shape
        
        self.model = Sequential([
            GRU(gru_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            GRU(gru_units, return_sequences=False),
            Dropout(dropout),
            Dense(dense_units, activation='relu'),
            Dense(1, activation='sigmoid' if self.model_type == 'classification' else 'linear')
        ])
        
        print(f"[OK] {self.name} model vytvorený s {gru_units} GRU jednotkami")


# =============================================================================
# MLP (MULTI-LAYER PERCEPTRON)
# =============================================================================

class MLPModel(BaseDeepModel):
    """
    Viacvrstvová neurónová sieť (MLP) pre štandardné features.
    
    MLP je základná feed-forward neurónová sieť vhodná pre
    tabulkové dáta bez časovej dimenzie.
    """
    
    def __init__(self, model_type: str = "regression"):
        """
        Inicializácia MLP modelu.
        
        Args:
            model_type (str): "classification" alebo "regression"
        """
        name = f"MLP ({'Klasifikácia' if model_type == 'classification' else 'Regresia'})"
        super().__init__(name, model_type)
        self.is_sequential = False
    
    def build_model(
        self,
        input_dim: int,
        hidden_layers: List[int] = None,
        dropout: float = 0.3,
        activation: str = 'relu'
    ) -> None:
        """
        Vytvorí MLP model.
        
        Args:
            input_dim (int): Dimenzia vstupu (počet features)
            hidden_layers (List[int], optional): Počet neurónov v skrytých vrstvách
            dropout (float): Dropout rate
            activation (str): Aktivačná funkcia
        """
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        self.input_shape = (input_dim,)
        
        model = Sequential()
        
        # Vstupná vrstva
        model.add(Dense(hidden_layers[0], activation=activation, input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        
        # Skryté vrstvy
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        
        # Výstupná vrstva
        if self.model_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(1, activation='linear'))
        
        self.model = model
        
        print(f"[OK] {self.name} model vytvorený")
        print(f"  - Vstupná dimenzia: {input_dim}")
        print(f"  - Skryté vrstvy: {hidden_layers}")


# =============================================================================
# CNN-LSTM HYBRID MODEL
# =============================================================================

class CNNLSTMModel(BaseDeepModel):
    """
    Hybridný CNN-LSTM model.
    
    Kombinuje konvolučné vrstvy pre extrakciu features
    s LSTM vrstvami pre zachytenie časových závislostí.
    """
    
    def __init__(self, model_type: str = "regression"):
        """
        Inicializácia CNN-LSTM modelu.
        
        Args:
            model_type (str): "classification" alebo "regression"
        """
        name = f"CNN-LSTM ({'Klasifikácia' if model_type == 'classification' else 'Regresia'})"
        super().__init__(name, model_type)
    
    def build_model(
        self,
        input_shape: Tuple[int, int],
        conv_filters: List[int] = None,
        lstm_units: int = 50,
        dropout: float = 0.2
    ) -> None:
        """
        Vytvorí CNN-LSTM model.
        
        Args:
            input_shape (Tuple[int, int]): Tvar vstupu
            conv_filters (List[int], optional): Počet filtrov v Conv vrstvách
            lstm_units (int): Počet LSTM jednotiek
            dropout (float): Dropout rate
        """
        if conv_filters is None:
            conv_filters = [64, 32]
        
        self.input_shape = input_shape
        
        model = Sequential()
        
        # Konvolučné vrstvy
        for i, filters in enumerate(conv_filters):
            if i == 0:
                model.add(Conv1D(filters, kernel_size=3, activation='relu', 
                                padding='same', input_shape=input_shape))
            else:
                model.add(Conv1D(filters, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2, padding='same'))
            model.add(Dropout(dropout))
        
        # LSTM vrstva
        model.add(LSTM(lstm_units, return_sequences=False))
        model.add(Dropout(dropout))
        
        # Dense vrstvy
        model.add(Dense(25, activation='relu'))
        
        # Výstupná vrstva
        if self.model_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(1, activation='linear'))
        
        self.model = model
        
        print(f"[OK] {self.name} model vytvorený")
        print(f"  - Vstupný tvar: {input_shape}")
        print(f"  - Conv filtre: {conv_filters}")
        print(f"  - LSTM jednotky: {lstm_units}")


# =============================================================================
# POMOCNÉ FUNKCIE
# =============================================================================

def prepare_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pripraví sekvencie pre rekurentné siete.
    
    Args:
        X (np.ndarray): Feature matica (n_samples, n_features)
        y (np.ndarray): Cieľový vektor
        sequence_length (int, optional): Dĺžka sekvencie
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Sekvencie X a y
    """
    if sequence_length is None:
        sequence_length = config.LSTM_PARAMS['sequence_length']
    
    X_seq = []
    y_seq = []
    
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)


def create_all_deep_learning_models(
    input_shape: Tuple[int, int],
    model_type: str = "classification",
    input_dim: int = None,
    quick: bool = False
) -> Dict[str, BaseDeepModel]:
    """
    Vytvorí všetky dostupné deep learning modely.
    
    V plnom režime vytvára: LSTM, GRU, CNN-LSTM a MLP (ak je zadaný input_dim).
    V rýchlom režime (quick=True) vytvára len: Simple LSTM a Simple GRU.
    
    Args:
        input_shape (Tuple[int, int]): Tvar vstupu pre sekvenčné modely
        model_type (str): "classification" alebo "regression"
        input_dim (int, optional): Dimenzia vstupu pre MLP (počet features)
        quick (bool): Rýchly mód – len jednoduché modely
        
    Returns:
        Dict[str, BaseDeepModel]: Slovník modelov
    """
    if not TENSORFLOW_AVAILABLE:
        print("[CHYBA] TensorFlow nie je nainštalovaný")
        return {}
    
    print("\n" + "=" * 60)
    print(f"VYTVÁRANIE DEEP LEARNING MODELOV ({model_type.upper()})")
    print("=" * 60)
    
    models = {}
    
    if quick:
        # Rýchly mód – len jednoduché varianty
        lstm = SimpleLSTMModel(model_type)
        lstm.build_model(input_shape)
        lstm.compile_model()
        models['Simple LSTM'] = lstm
        
        gru = SimpleGRUModel(model_type)
        gru.build_model(input_shape)
        gru.compile_model()
        models['Simple GRU'] = gru
    else:
        # Plný mód – LSTM, GRU, CNN-LSTM a MLP
        lstm = LSTMModel(model_type)
        lstm.build_model(input_shape)
        lstm.compile_model()
        models['LSTM'] = lstm
        
        gru = GRUModel(model_type)
        gru.build_model(input_shape)
        gru.compile_model()
        models['GRU'] = gru
        
        cnn_lstm = CNNLSTMModel(model_type)
        cnn_lstm.build_model(input_shape)
        cnn_lstm.compile_model()
        models['CNN-LSTM'] = cnn_lstm
        
        if input_dim is not None:
            mlp = MLPModel(model_type)
            mlp.build_model(input_dim)
            mlp.compile_model()
            models['MLP'] = mlp
    
    print(f"\n[OK] Vytvorených {len(models)} deep learning modelov")
    
    return models


def create_mlp_model(
    input_dim: int,
    model_type: str = "classification"
) -> MLPModel:
    """
    Vytvorí MLP model pre štandardné (nesekvenčné) dáta.
    
    Args:
        input_dim (int): Počet vstupných features
        model_type (str): "classification" alebo "regression"
        
    Returns:
        MLPModel: Vytvorený a skompilovaný model
    """
    mlp = MLPModel(model_type)
    mlp.build_model(input_dim)
    mlp.compile_model()
    
    return mlp


def train_deep_model(
    model: BaseDeepModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = None,
    batch_size: int = None
) -> Dict:
    """
    Natrénuje deep learning model.
    
    Args:
        model (BaseDeepModel): Model na trénovanie
        X_train (np.ndarray): Trénovacie dáta
        y_train (np.ndarray): Trénovacie labels
        X_val (np.ndarray, optional): Validačné dáta
        y_val (np.ndarray, optional): Validačné labels
        epochs (int, optional): Počet epoch
        batch_size (int, optional): Veľkosť batch
        
    Returns:
        Dict: História trénovania
    """
    return model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping=True
    )


# =============================================================================
# HLAVNÁ FUNKCIA PRE TESTOVANIE MODULU
# =============================================================================

if __name__ == "__main__":
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow nie je dostupný. Testovanie preskočené.")
    else:
        import data_downloader
        import data_preprocessing
        import feature_engineering
        
        print("=" * 60)
        print("TESTOVANIE MODULU MODELS_DEEP_LEARNING")
        print("=" * 60)
        
        # Načítanie dát
        try:
            df = data_downloader.load_stock_data("AAPL")
        except FileNotFoundError:
            df = data_downloader.download_stock_data("AAPL")
        
        df_processed = data_preprocessing.preprocess_pipeline(df, ticker="AAPL", save_to_file=False)
        df_features = feature_engineering.create_all_features(df_processed, ticker="AAPL", save_to_file=False)
        
        # Príprava dát
        feature_cols = feature_engineering.get_feature_list(df_features)
        target_col = 'Target_Direction_1d'
        
        split_data = data_preprocessing.split_data(
            df_features,
            target_column=target_col,
            feature_columns=feature_cols
        )
        
        X_train = split_data['X_train']
        y_train = split_data['y_train']
        X_val = split_data['X_val']
        y_val = split_data['y_val']
        
        # Príprava sekvencií
        sequence_length = 30
        X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, sequence_length)
        
        print(f"\nTvar sekvenčných dát: {X_train_seq.shape}")
        
        # Test Simple LSTM
        print("\n--- Test Simple LSTM ---")
        lstm = SimpleLSTMModel("classification")
        lstm.build_model(input_shape=(sequence_length, X_train.shape[1]))
        lstm.compile_model()
        
        # Rýchle trénovanie pre test (5 epoch)
        history = lstm.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=5)
        
        # Predikcia
        y_pred = lstm.predict(X_val_seq)
        accuracy = (y_pred == y_val_seq).mean()
        print(f"\n[OK] Simple LSTM validačná accuracy: {accuracy:.4f}")
        
        # Test MLP
        print("\n--- Test MLP ---")
        mlp = create_mlp_model(X_train.shape[1], "classification")
        mlp.train(X_train, y_train, X_val, y_val, epochs=5)
        
        y_pred_mlp = mlp.predict(X_val)
        accuracy_mlp = (y_pred_mlp == y_val).mean()
        print(f"[OK] MLP validačná accuracy: {accuracy_mlp:.4f}")
