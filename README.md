# Strojové učenie a simulácia pri modelovaní finančných trhov
## Hybridný prístup - Diplomová práca

Tento projekt implementuje komplexný systém pre analýzu finančných trhov pomocou strojového učenia a simulačných metód.

---

## 📁 Štruktúra projektu

```
financial_ml/
├── config.py               # Konfigurácia a parametre projektu
├── data_downloader.py      # Sťahovanie dát z Yahoo Finance
├── data_preprocessing.py   # Predspracovanie a čistenie dát
├── feature_engineering.py  # Technické indikátory a features
├── visualization.py        # Vizualizácia dát a výsledkov
├── models_classical.py     # Klasické ML modely (RF, XGBoost, SVM, ...)
├── models_deep_learning.py # Deep learning modely (LSTM, GRU, MLP)
├── model_evaluation.py     # Vyhodnotenie a porovnanie modelov
├── simulation.py           # Monte Carlo simulácia a backtesting
├── main.py                 # Hlavný spúšťací skript
├── requirements.txt        # Závislosti projektu
├── README.md               # Tento súbor
├── data/                   # Adresár pre dáta (vytvorí sa automaticky)
├── models/                 # Adresár pre uložené modely
├── results/                # Adresár pre výsledky
└── plots/                  # Adresár pre grafy
```

---

## 🚀 Inštalácia

### 1. Vytvorenie virtuálneho prostredia (odporúčané)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Inštalácia závislostí

```bash
pip install -r requirements.txt
```

---

## 💻 Použitie

### Základné spustenie (kompletný pipeline)

```bash
python main.py
```

### Analýza konkrétnej akcie

```bash
python main.py --ticker AAPL
```

### Analýza viacerých akcií

```bash
python main.py --tickers AAPL MSFT GOOGL
```

### Rýchly mód (menej modelov, kratšie trénovanie)

```bash
python main.py --quick
```

### Preskočenie sťahovania (použitie lokálnych dát)

```bash
python main.py --skip-download
```

### Všetky dostupné argumenty

```bash
python main.py --help
```

| Argument | Skratka | Popis |
|----------|---------|-------|
| `--ticker` | `-t` | Konkrétny ticker na analýzu |
| `--tickers` | | Zoznam tickerov |
| `--skip-download` | | Použiť lokálne dáta |
| `--quick` | `-q` | Rýchly mód |
| `--skip-visualization` | | Preskočiť vizualizácie |
| `--skip-dl` | | Preskočiť deep learning |
| `--skip-simulation` | | Preskočiť simulácie |

---

## 📊 Funkcionality

### 1. Sťahovanie a analýza dát
- Automatické sťahovanie z Yahoo Finance
- Detekcia a ošetrenie outlierov (IQR, Z-score)
- Výpočet výnosov a volatility
- Spracovanie chýbajúcich hodnôt

### 2. Feature Engineering
- **Kĺzavé priemery**: SMA, EMA (rôzne periódy)
- **Oscilátory**: RSI, Stochastic, Williams %R
- **Trendy**: MACD, ADX
- **Volatilita**: Bollinger Bands, ATR
- **Volume**: OBV, PVT, AD Line
- **Momentum**: ROC, CCI
- **Časové features**: Deň v týždni, mesiac, štvrťrok

### 3. Klasické ML modely
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Logistická regresia
- Gradient Boosting
- Rozhodovací strom
- AdaBoost
- Ridge/Lasso regresia

### 4. Deep Learning modely
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- MLP (Multi-Layer Perceptron)
- CNN-LSTM hybrid

### 5. Vyhodnotenie modelov
- **Klasifikácia**: Accuracy, Precision, Recall, F1, AUC-ROC
- **Regresia**: MSE, RMSE, MAE, R², MAPE
- **Finančné metriky**: Sharpe Ratio, Max Drawdown, Win Rate
- Automatické porovnanie všetkých modelov

### 6. Simulácie
- **Monte Carlo**: Geometrický Brownov pohyb (GBM)
- **Rizikové metriky**: VaR, CVaR/Expected Shortfall
- **Backtesting**: MA crossover, RSI stratégie
- **Vlastné stratégie**: Na základe ML predikcií

---

## ⚙️ Konfigurácia

Všetky parametre je možné upraviť v súbore `config.py`:

```python
# Zoznam akcií na analýzu
STOCK_TICKERS = ["AAPL", "MSFT", "GOOGL", ...]

# Časové obdobie
START_DATE = "2019-01-01"
END_DATE = "2024-01-01"

# Parametre modelov
RF_PARAMS = {...}
XGB_PARAMS = {...}
LSTM_PARAMS = {...}

# Parametre simulácií
MONTE_CARLO_SIMULATIONS = 1000
INITIAL_CAPITAL = 100000
```

---

## 📈 Výstupy

Po spustení sa vytvoria:

1. **`data/`** - Stiahnuté a predspracované dáta (CSV)
2. **`models/`** - Natrénované modely (joblib, keras)
3. **`plots/`** - Vizualizácie:
   - Cenové grafy
   - Technické indikátory
   - Distribúcia výnosov
   - Korelačné matice
   - Monte Carlo simulácie
   - Backtest výsledky
4. **`results/`** - Výsledky a reporty:
   - Porovnanie modelov (CSV, JSON)
   - Záverečný report (TXT)

---

## 🔧 Riešenie problémov

### TensorFlow sa nedá nainštalovať
Program funguje aj bez TensorFlow - deep learning modely budú jednoducho preskočené.

### Chyba pri sťahovaní dát
Skontrolujte internetové pripojenie a správnosť ticker symbolov.

### Nedostatok pamäte
Použite `--quick` mód alebo analyzujte menej tickerov naraz.

---

## 📚 Citácia

Ak používate tento kód vo svojej práci, prosím citujte:

```
Diplomová práca: Strojové učenie a simulácia pri modelovaní finančných trhov - Hybridný prístup
```

---

## 📝 Licencia

Tento projekt bol vytvorený pre akademické účely v rámci diplomovej práce.
