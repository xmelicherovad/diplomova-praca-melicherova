# -*- coding: utf-8 -*-
"""
Modul pre vyhodnotenie a porovnanie modelov.

Tento modul poskytuje funkcie na:
- Výpočet metrík pre klasifikáciu (accuracy, precision, recall, F1, AUC-ROC)
- Výpočet metrík pre regresiu (MSE, RMSE, MAE, R², MAPE)
- Porovnanie viacerých modelov
- Generovanie reportov
- Štatistické testy

Autor: Dominika Melicherová
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Sklearn metriky
from sklearn.metrics import (
    # Klasifikačné metriky
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    # Regresné metriky
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

# Import konfigurácie a vizualizácie
import config
import visualization as viz


# =============================================================================
# KLASIFIKAČNÉ METRIKY
# =============================================================================

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Vypočíta všetky relevantné klasifikačné metriky.
    
    Args:
        y_true (np.ndarray): Skutočné triedy
        y_pred (np.ndarray): Predikované triedy
        y_pred_proba (np.ndarray, optional): Pravdepodobnosti predikcie
        
    Returns:
        Dict[str, float]: Slovník s metrikami
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # AUC-ROC ak máme pravdepodobnosti
    if y_pred_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics['auc_roc'] = None
    
    # Konfúzna matica
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # True/False Positive/Negative rates
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> str:
    """
    Vypíše podrobný klasifikačný report.
    
    Args:
        y_true (np.ndarray): Skutočné triedy
        y_pred (np.ndarray): Predikované triedy
        model_name (str): Názov modelu
        
    Returns:
        str: Textový report
    """
    print("\n" + "=" * 60)
    print(f"KLASIFIKAČNÝ REPORT - {model_name}")
    print("=" * 60)
    
    # Základné metriky
    metrics = calculate_classification_metrics(y_true, y_pred)
    
    print(f"\nZákladné metriky:")
    print(f"  Presnosť (Accuracy):  {metrics['accuracy']:.4f}")
    print(f"  Precíznosť (Precision): {metrics['precision']:.4f}")
    print(f"  Úplnosť (Recall):     {metrics['recall']:.4f}")
    print(f"  F1 skóre:             {metrics['f1_score']:.4f}")
    
    if 'auc_roc' in metrics and metrics['auc_roc'] is not None:
        print(f"  AUC-ROC:              {metrics['auc_roc']:.4f}")
    
    # Sklearn classification report
    print("\nPodrobný report:")
    target_names = ['Pokles (0)', 'Rast (1)']
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    
    # Konfúzna matica
    print("Konfúzna matica:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Skutočný Pokles, Predikovaný Pokles (TN): {cm[0, 0]}")
    print(f"  Skutočný Pokles, Predikovaný Rast (FP):   {cm[0, 1]}")
    print(f"  Skutočný Rast, Predikovaný Pokles (FN):   {cm[1, 0]}")
    print(f"  Skutočný Rast, Predikovaný Rast (TP):     {cm[1, 1]}")
    
    print("=" * 60)
    
    return report


# =============================================================================
# REGRESNÉ METRIKY
# =============================================================================

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Vypočíta všetky relevantné regresné metriky.
    
    Args:
        y_true (np.ndarray): Skutočné hodnoty
        y_pred (np.ndarray): Predikované hodnoty
        
    Returns:
        Dict[str, float]: Slovník s metrikami
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }
    
    # MAPE - opatrne s delením nulou
    try:
        # Filtrovanie nulových hodnôt
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['mape'] = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
        else:
            metrics['mape'] = None
    except Exception:
        metrics['mape'] = None
    
    # Dodatočné metriky
    # Smerová presnosť (či predikcia správne predpovedá smer zmeny)
    if len(y_true) > 1:
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        metrics['directional_accuracy'] = (actual_direction == pred_direction).mean()
    
    # Maximum error
    metrics['max_error'] = np.max(np.abs(y_true - y_pred))
    
    return metrics


def print_regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Vypíše podrobný regresný report.
    
    Args:
        y_true (np.ndarray): Skutočné hodnoty
        y_pred (np.ndarray): Predikované hodnoty
        model_name (str): Názov modelu
        
    Returns:
        Dict[str, float]: Slovník s metrikami
    """
    print("\n" + "=" * 60)
    print(f"REGRESNÝ REPORT - {model_name}")
    print("=" * 60)
    
    metrics = calculate_regression_metrics(y_true, y_pred)
    
    print(f"\nChybové metriky:")
    print(f"  MSE (Mean Squared Error):        {metrics['mse']:.6f}")
    print(f"  RMSE (Root Mean Squared Error):  {metrics['rmse']:.6f}")
    print(f"  MAE (Mean Absolute Error):       {metrics['mae']:.6f}")
    print(f"  Max Error:                       {metrics['max_error']:.6f}")
    
    if metrics['mape'] is not None:
        print(f"  MAPE (Mean Abs. Pct. Error):     {metrics['mape']:.2f}%")
    
    print(f"\nKvalita modelu:")
    print(f"  R² (Coefficient of Determination): {metrics['r2']:.6f}")
    
    if 'directional_accuracy' in metrics:
        print(f"  Smerová presnosť:                  {metrics['directional_accuracy']:.4f}")
    
    # Interpretácia R²
    r2 = metrics['r2']
    if r2 >= 0.9:
        interpretation = "Výborná zhoda"
    elif r2 >= 0.7:
        interpretation = "Dobrá zhoda"
    elif r2 >= 0.5:
        interpretation = "Prijateľná zhoda"
    elif r2 >= 0.3:
        interpretation = "Slabá zhoda"
    else:
        interpretation = "Veľmi slabá zhoda"
    
    print(f"  Interpretácia R²: {interpretation}")
    
    print("=" * 60)
    
    return metrics


# =============================================================================
# POROVNANIE MODELOV
# =============================================================================

class ModelComparator:
    """
    Trieda pre porovnávanie viacerých modelov.
    
    Umožňuje systematické vyhodnotenie a porovnanie výkonnosti
    rôznych modelov na rovnakých dátach.
    """
    
    def __init__(self, model_type: str = "classification"):
        """
        Inicializácia komparátora.
        
        Args:
            model_type (str): "classification" alebo "regression"
        """
        self.model_type = model_type
        self.results = {}
        self.best_model = None
        self.primary_metric = 'accuracy' if model_type == 'classification' else 'rmse'
    
    def add_result(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
        additional_info: Dict = None
    ) -> Dict[str, float]:
        """
        Pridá výsledky modelu do porovnania.
        
        Args:
            model_name (str): Názov modelu
            y_true (np.ndarray): Skutočné hodnoty
            y_pred (np.ndarray): Predikované hodnoty
            y_pred_proba (np.ndarray, optional): Pravdepodobnosti
            additional_info (Dict, optional): Dodatočné informácie
            
        Returns:
            Dict[str, float]: Metriky modelu
        """
        if self.model_type == "classification":
            metrics = calculate_classification_metrics(y_true, y_pred, y_pred_proba)
        else:
            metrics = calculate_regression_metrics(y_true, y_pred)
        
        # Pridanie dodatočných informácií
        if additional_info:
            metrics.update(additional_info)
        
        self.results[model_name] = metrics
        
        print(f"[OK] Výsledky pre '{model_name}' pridané")
        
        return metrics
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Vytvorí tabuľku porovnania modelov.
        
        Returns:
            pd.DataFrame: Tabuľka s metrikami všetkých modelov
        """
        if not self.results:
            print("[VAROVANIE] Žiadne výsledky na porovnanie")
            return pd.DataFrame()
        
        # Výber numerických metrík
        all_metrics = set()
        for metrics in self.results.values():
            all_metrics.update([k for k, v in metrics.items() 
                               if isinstance(v, (int, float)) and v is not None])
        
        # Vytvorenie DataFrame
        data = []
        for model_name, metrics in self.results.items():
            row = {'Model': model_name}
            for metric in all_metrics:
                row[metric] = metrics.get(metric, None)
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.set_index('Model')
        
        return df
    
    def find_best_model(self, metric: str = None) -> Tuple[str, float]:
        """
        Nájde najlepší model podľa zadanej metriky.
        
        Args:
            metric (str, optional): Metrika pre porovnanie
            
        Returns:
            Tuple[str, float]: Názov najlepšieho modelu a jeho skóre
        """
        if metric is None:
            metric = self.primary_metric
        
        # Metriky kde nižšia hodnota je lepšia
        lower_is_better = ['mse', 'rmse', 'mae', 'mape', 'max_error', 
                          'false_positive_rate', 'false_negative_rate']
        
        best_model = None
        best_score = None
        
        for model_name, metrics in self.results.items():
            score = metrics.get(metric)
            if score is None:
                continue
            
            if best_score is None:
                best_score = score
                best_model = model_name
            elif metric in lower_is_better:
                if score < best_score:
                    best_score = score
                    best_model = model_name
            else:
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        self.best_model = best_model
        
        return best_model, best_score
    
    def print_comparison(self) -> None:
        """Vypíše porovnanie všetkých modelov."""
        print("\n" + "=" * 80)
        print("POROVNANIE MODELOV")
        print("=" * 80)
        
        df = self.get_comparison_table()
        
        if df.empty:
            print("Žiadne výsledky na porovnanie.")
            return
        
        # Formátovanie čísel
        pd.set_option('display.float_format', lambda x: f'{x:.4f}')
        print(df.to_string())
        
        # Najlepší model
        best_model, best_score = self.find_best_model()
        print(f"\n{'='*80}")
        print(f"NAJLEPŠÍ MODEL podľa {self.primary_metric}: {best_model} ({best_score:.4f})")
        print("=" * 80)
    
    def generate_report(
        self,
        output_path: str = None,
        include_plots: bool = True
    ) -> str:
        """
        Vygeneruje kompletný report porovnania.
        
        Args:
            output_path (str, optional): Cesta pre uloženie reportu
            include_plots (bool): Zahrnúť grafy
            
        Returns:
            str: Cesta k reportu
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(config.RESULTS_DIR, f"model_comparison_{timestamp}")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Tabuľka porovnania
        df = self.get_comparison_table()
        df.to_csv(os.path.join(output_path, "comparison_table.csv"))
        
        # JSON s detailmi
        results_json = {}
        for model_name, metrics in self.results.items():
            results_json[model_name] = {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in metrics.items()
            }
        
        with open(os.path.join(output_path, "results_detail.json"), 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        
        # Grafy
        if include_plots and self.results:
            # Graf porovnania hlavnej metriky
            viz.plot_model_comparison(
                self.results,
                metric=self.primary_metric,
                title=f"Porovnanie modelov - {self.primary_metric}",
                save_path=os.path.join(output_path, f"comparison_{self.primary_metric}.png"),
                model_type=self.model_type
            )
        
        print(f"[OK] Report uložený do {output_path}")
        
        return output_path
    
    def rank_models(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Zoradí modely podľa viacerých metrík.
        
        Args:
            metrics (List[str], optional): Zoznam metrík pre hodnotenie
            
        Returns:
            pd.DataFrame: Tabuľka s poradiami
        """
        if metrics is None:
            if self.model_type == "classification":
                metrics = ['accuracy', 'f1_score', 'precision', 'recall']
            else:
                metrics = ['rmse', 'mae', 'r2']
        
        lower_is_better = ['mse', 'rmse', 'mae', 'mape', 'max_error']
        
        rankings = {}
        
        for metric in metrics:
            if metric not in self.get_comparison_table().columns:
                continue
            
            scores = []
            for model_name, model_metrics in self.results.items():
                score = model_metrics.get(metric)
                if score is not None:
                    scores.append((model_name, score))
            
            # Zoradenie
            ascending = metric in lower_is_better
            scores.sort(key=lambda x: x[1], reverse=not ascending)
            
            # Priradenie poradia
            for rank, (model_name, _) in enumerate(scores, 1):
                if model_name not in rankings:
                    rankings[model_name] = {}
                rankings[model_name][metric] = rank
        
        # Vytvorenie DataFrame
        df = pd.DataFrame(rankings).T
        df['Priemerné poradie'] = df.mean(axis=1)
        df = df.sort_values('Priemerné poradie')
        
        return df


# =============================================================================
# FINANČNÉ METRIKY
# =============================================================================

def calculate_financial_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: np.ndarray = None
) -> Dict[str, float]:
    """
    Vypočíta finančne relevantné metriky.
    
    Args:
        y_true (np.ndarray): Skutočné smery/hodnoty
        y_pred (np.ndarray): Predikované smery/hodnoty
        returns (np.ndarray, optional): Skutočné denné výnosy v percentách
            (napr. 1.5 znamená +1.5 %)
        
    Returns:
        Dict[str, float]: Finančné metriky
    """
    metrics = {}
    
    # Smerová presnosť
    metrics['directional_accuracy'] = float((y_true == y_pred).mean())
    
    # Ak máme výnosy, vypočítame stratégiu
    if returns is not None:
        # Konverzia na desatinné čísla (1.5 % → 0.015)
        ret_decimal = returns / 100.0

        # Stratégia: Ak predikujeme rast (1), investujeme, inak nie
        strategy_returns = ret_decimal * np.where(y_pred == 1, 1, 0)
        
        # Kumulatívny výnos
        metrics['cumulative_return'] = float((1 + strategy_returns).prod() - 1)
        
        # Priemerný denný výnos
        metrics['mean_return'] = float(strategy_returns.mean())
        
        # Sharpe Ratio (s risk-free rate)
        daily_rf = config.RISK_FREE_RATE / 252
        excess = strategy_returns - daily_rf
        if excess.std() > 0:
            metrics['sharpe_ratio'] = float(
                (excess.mean() / excess.std()) * np.sqrt(252)
            )
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Maximum Drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - rolling_max) / rolling_max
        metrics['max_drawdown'] = float(drawdown.min())
        
        # Win rate
        active = strategy_returns[strategy_returns != 0]
        if len(active) > 0:
            metrics['win_rate'] = float((active > 0).sum() / len(active))
        else:
            metrics['win_rate'] = 0.0

        # Profit factor
        gains = strategy_returns[strategy_returns > 0].sum()
        losses = abs(strategy_returns[strategy_returns < 0].sum())
        metrics['profit_factor'] = float(gains / losses) if losses > 0 else float('inf')
    
    return metrics


def print_financial_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Vypíše finančný report modelu.
    
    Args:
        y_true (np.ndarray): Skutočné smery
        y_pred (np.ndarray): Predikované smery
        returns (np.ndarray): Skutočné výnosy
        model_name (str): Názov modelu
        
    Returns:
        Dict[str, float]: Finančné metriky
    """
    print("\n" + "=" * 60)
    print(f"FINANČNÝ REPORT - {model_name}")
    print("=" * 60)
    
    metrics = calculate_financial_metrics(y_true, y_pred, returns)
    
    print(f"\nVýkonnosť stratégie:")
    print(f"  Smerová presnosť:      {metrics['directional_accuracy']:.2%}")
    print(f"  Kumulatívny výnos:     {metrics['cumulative_return']:.2%}")
    print(f"  Priemerný denný výnos: {metrics['mean_return']:.6f}")
    print(f"  Sharpe Ratio (ročný):  {metrics['sharpe_ratio']:.4f}")
    print(f"  Maximum Drawdown:      {metrics['max_drawdown']:.2%}")
    print(f"  Win Rate:              {metrics['win_rate']:.2%}")
    if 'profit_factor' in metrics:
        pf = metrics['profit_factor']
        print(f"  Profit Factor:         {pf:.4f}" if pf != float('inf') else "  Profit Factor:         ∞")
    
    print("=" * 60)
    
    return metrics


# =============================================================================
# ŠTATISTICKÉ TESTY
# =============================================================================

def compare_models_statistical(
    results1: np.ndarray,
    results2: np.ndarray,
    test_type: str = "ttest"
) -> Dict[str, float]:
    """
    Vykoná štatistický test na porovnanie dvoch modelov.
    
    Args:
        results1 (np.ndarray): Výsledky prvého modelu (napr. z krížovej validácie)
        results2 (np.ndarray): Výsledky druhého modelu
        test_type (str): Typ testu - "ttest" alebo "wilcoxon"
        
    Returns:
        Dict[str, float]: Výsledky testu
    """
    from scipy import stats
    
    if test_type == "ttest":
        # Párový t-test
        t_stat, p_value = stats.ttest_rel(results1, results2)
        test_name = "Párový t-test"
    elif test_type == "wilcoxon":
        # Wilcoxonov test (neparametrický)
        t_stat, p_value = stats.wilcoxon(results1, results2)
        test_name = "Wilcoxonov test"
    else:
        raise ValueError(f"Neznámy typ testu: {test_type}")
    
    # Interpretácia
    alpha = 0.05
    is_significant = p_value < alpha
    
    result = {
        'test': test_name,
        'statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'alpha': alpha,
        'interpretation': "Modely sa štatisticky významne líšia" if is_significant 
                         else "Modely sa štatisticky významne nelíšia"
    }
    
    print(f"\n{test_name}:")
    print(f"  Štatistika: {t_stat:.4f}")
    print(f"  P-hodnota: {p_value:.4f}")
    print(f"  Záver (α={alpha}): {result['interpretation']}")
    
    return result


# =============================================================================
# POMOCNÉ FUNKCIE
# =============================================================================

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = None,
    model_type: str = "classification",
    returns: np.ndarray = None,
    print_report: bool = True
) -> Dict[str, Any]:
    """
    Kompletné vyhodnotenie modelu.
    
    Args:
        model: Natrénovaný model
        X_test (np.ndarray): Testovacie features
        y_test (np.ndarray): Testovacie labels
        model_name (str, optional): Názov modelu
        model_type (str): "classification" alebo "regression"
        returns (np.ndarray, optional): Skutočné výnosy pre finančné metriky
        print_report (bool): Vypísať report
        
    Returns:
        Dict[str, Any]: Všetky metriky
    """
    if model_name is None:
        model_name = model.__class__.__name__ if hasattr(model, '__class__') else "Model"
    
    # Predikcia
    y_pred = model.predict(X_test)
    
    # Pravdepodobnosti pre klasifikáciu
    y_pred_proba = None
    if model_type == "classification" and hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_test)
            if len(y_pred_proba.shape) > 1:
                y_pred_proba = y_pred_proba[:, 1]
        except Exception:
            pass
    
    # Výpočet metrík
    if model_type == "classification":
        metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        if print_report:
            print_classification_report(y_test, y_pred, model_name)
    else:
        metrics = calculate_regression_metrics(y_test, y_pred)
        if print_report:
            print_regression_report(y_test, y_pred, model_name)
    
    # Finančné metriky
    if returns is not None and model_type == "classification":
        financial_metrics = calculate_financial_metrics(y_test, y_pred, returns)
        metrics['financial'] = financial_metrics
        if print_report:
            print_financial_report(y_test, y_pred, returns, model_name)
    
    # Pridanie predikcií
    metrics['y_pred'] = y_pred
    metrics['y_pred_proba'] = y_pred_proba
    
    return metrics


def save_evaluation_results(
    results: Dict[str, Dict],
    output_path: str = None
) -> str:
    """
    Uloží výsledky vyhodnotenia do súborov.
    
    Args:
        results (Dict[str, Dict]): Slovník s výsledkami modelov
        output_path (str, optional): Cesta pre uloženie
        
    Returns:
        str: Cesta k výsledkom
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(config.RESULTS_DIR, f"evaluation_{timestamp}")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Konverzia na serializovateľný formát
    serializable_results = {}
    for model_name, metrics in results.items():
        serializable_results[model_name] = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_results[model_name][key] = value.tolist()
            elif isinstance(value, (np.int64, np.float64)):
                serializable_results[model_name][key] = float(value)
            elif isinstance(value, dict):
                serializable_results[model_name][key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[model_name][key] = value
    
    # JSON
    json_path = os.path.join(output_path, "evaluation_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Výsledky uložené do {output_path}")
    
    return output_path


# =============================================================================
# HLAVNÁ FUNKCIA PRE TESTOVANIE MODULU
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTOVANIE MODULU MODEL_EVALUATION")
    print("=" * 60)
    
    # Simulované dáta pre test
    np.random.seed(config.RANDOM_SEED)
    n_samples = 500
    
    # Klasifikačné dáta
    y_true_clf = np.random.randint(0, 2, n_samples)
    y_pred_clf = y_true_clf.copy()
    # Pridanie šumu
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    y_pred_clf[noise_idx] = 1 - y_pred_clf[noise_idx]
    y_pred_proba = np.random.uniform(0.3, 0.7, n_samples)
    y_pred_proba[y_pred_clf == 1] = np.random.uniform(0.5, 0.9, (y_pred_clf == 1).sum())
    
    # Regresné dáta
    y_true_reg = np.random.randn(n_samples) * 10 + 100
    y_pred_reg = y_true_reg + np.random.randn(n_samples) * 2
    
    # Test klasifikačných metrík
    print("\n--- Test klasifikačných metrík ---")
    clf_metrics = calculate_classification_metrics(y_true_clf, y_pred_clf, y_pred_proba)
    print_classification_report(y_true_clf, y_pred_clf, "Test Model")
    
    # Test regresných metrík
    print("\n--- Test regresných metrík ---")
    reg_metrics = calculate_regression_metrics(y_true_reg, y_pred_reg)
    print_regression_report(y_true_reg, y_pred_reg, "Test Model")
    
    # Test komparátora
    print("\n--- Test komparátora modelov ---")
    comparator = ModelComparator("classification")
    
    # Pridanie simulovaných výsledkov
    comparator.add_result("Model A", y_true_clf, y_pred_clf, y_pred_proba)
    
    # Druhý "model" s trochu inými výsledkami
    y_pred_clf2 = y_true_clf.copy()
    noise_idx2 = np.random.choice(n_samples, size=int(n_samples * 0.25), replace=False)
    y_pred_clf2[noise_idx2] = 1 - y_pred_clf2[noise_idx2]
    comparator.add_result("Model B", y_true_clf, y_pred_clf2)
    
    comparator.print_comparison()
    
    # Poradie modelov
    print("\nPoradie modelov:")
    print(comparator.rank_models())
    
    print("\n[OK] Testovanie modulu dokončené")
