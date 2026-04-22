# -*- coding: utf-8 -*-
"""
Modul pre simulácie a backtesting.

Tento modul poskytuje funkcie na:
- Monte Carlo simuláciu cenových pohybov
- Backtesting obchodných stratégií založených na modeloch
- Výpočet rizikových metrík (VaR, CVaR)
- Analýzu citlivosti

Autor: Diplomová práca - Strojové učenie a simulácia pri modelovaní finančných trhov
"""

import os
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime
import warnings

# Import konfigurácie a vizualizácie
import config
import visualization as viz


# =============================================================================
# FITOVANIE ROZDELENIA VÝNOSOV
# =============================================================================

class ReturnDistributionFitter:
    """
    Trieda pre identifikáciu a fitovanie rozdelenia logaritmických výnosov.

    Testuje viacero kandidátnych rozdelení pomocou maximálnej vierohodnosti
    a vyberie najlepšie na základe AIC a Kolmogorov-Smirnov testu.
    Toto je jadro hybridného prístupu - namiesto predpokladu normálneho
    rozdelenia (GBM) identifikujeme skutočné rozdelenie výnosov.
    """

    CANDIDATE_DISTRIBUTIONS = {
        'norm':      'Normálne (Gaussovo) rozdelenie',
        't':         'Studentovo t-rozdelenie',
        'skewnorm':  'Skreslené normálne rozdelenie',
        'laplace':   'Laplaceovo rozdelenie',
        'johnsonsu': 'Johnson SU rozdelenie',
        'logistic':  'Logistické rozdelenie',
        'gennorm':   'Zovšeobecnené normálne rozdelenie',
        'nct':       'Necentralizované t-rozdelenie',
    }

    def __init__(self, returns: np.ndarray):
        """
        Inicializácia fitteru.

        Args:
            returns (np.ndarray): Logaritmické výnosy (z np.diff(np.log(prices)))
        """
        self.returns = np.asarray(returns, dtype=float)
        self.returns = self.returns[np.isfinite(self.returns)]
        self.best_distribution: Optional[str] = None
        self.best_params: Optional[tuple] = None
        self.all_results: Dict[str, Any] = {}
        self.fit_complete: bool = False

    def fit(self, distributions: List[str] = None) -> Dict[str, Any]:
        """
        Nafituje kandidátske distribúcie a vyberie najlepšiu podľa AIC.

        Args:
            distributions (List[str], optional): Zoznam scipy.stats názvov distribúcií.
                Ak None, použijú sa všetky CANDIDATE_DISTRIBUTIONS.

        Returns:
            Dict: Výsledky pre každú distribúciu (parametre, AIC, KS p-hodnota, ...)
        """
        if distributions is None:
            distributions = list(self.CANDIDATE_DISTRIBUTIONS.keys())

        print(f"[INFO] Fitovanie {len(distributions)} rozdelení na výnosoch...")

        results = {}
        n = len(self.returns)

        for dist_name in distributions:
            try:
                dist = getattr(scipy_stats, dist_name)
                params = dist.fit(self.returns)

                # Log-vierohodnosť → AIC / BIC
                log_lik = float(np.sum(dist.logpdf(self.returns, *params)))
                if not np.isfinite(log_lik):
                    continue
                k = len(params)
                aic = 2 * k - 2 * log_lik
                bic = k * np.log(n) - 2 * log_lik

                # Kolmogorov-Smirnov test
                ks_stat, ks_pval = scipy_stats.kstest(
                    self.returns, dist_name, args=params
                )

                results[dist_name] = {
                    'params':        params,
                    'log_likelihood': log_lik,
                    'aic':           aic,
                    'bic':           bic,
                    'ks_statistic':  ks_stat,
                    'ks_pvalue':     ks_pval,
                    'name_sk':       self.CANDIDATE_DISTRIBUTIONS.get(dist_name, dist_name),
                }

            except Exception as e:
                print(f"  [VAROVANIE] Fitovanie '{dist_name}' zlyhalo: {e}")

        self.all_results = results

        if results:
            self.best_distribution = min(results, key=lambda k: results[k]['aic'])
            self.best_params = results[self.best_distribution]['params']
            self.fit_complete = True

        self._print_results()
        return results

    def _print_results(self) -> None:
        """Vypíše tabuľku výsledkov fitu."""
        print(f"\n{'='*65}")
        print("VÝSLEDKY FITU ROZDELENIA VÝNOSOV")
        print(f"{'='*65}")
        print(f"{'Rozdelenie':<38} {'AIC':>10} {'KS p-hodnota':>13}")
        print("-" * 65)

        sorted_res = sorted(self.all_results.items(), key=lambda x: x[1]['aic'])
        for dist_name, res in sorted_res:
            marker = " <- NAJLEPŠIE" if dist_name == self.best_distribution else ""
            print(
                f"{res['name_sk']:<38} {res['aic']:>10.2f}"
                f" {res['ks_pvalue']:>13.4f}{marker}"
            )

        if self.best_distribution:
            best = self.all_results[self.best_distribution]
            print(f"\nNajlepšie rozdelenie : {best['name_sk']}")
            print(f"Parametre            : {self.best_params}")
            print(f"Log-vierohodnosť     : {best['log_likelihood']:.4f}")
            print(f"AIC                  : {best['aic']:.4f}")
        print(f"{'='*65}")

    def generate_samples(self, n_samples: int, seed: int = None) -> np.ndarray:
        """
        Vygeneruje náhodné vzorky z nafitovanej distribúcie.

        Args:
            n_samples (int): Počet vzoriek
            seed (int, optional): Random seed

        Returns:
            np.ndarray: Vzorky z nafitovanej distribúcie
        """
        if not self.fit_complete:
            raise ValueError("Najprv spustite metódu fit().")

        if seed is not None:
            np.random.seed(seed)

        dist = getattr(scipy_stats, self.best_distribution)
        return dist.rvs(*self.best_params, size=n_samples)

    def get_empirical_moments(self) -> Dict[str, float]:
        """
        Vráti empirické štatistické momenty výnosov.

        Returns:
            Dict: Priemer, std, šikmosť, špicatosť, min, max
        """
        return {
            'mean':     float(np.mean(self.returns)),
            'std':      float(np.std(self.returns)),
            'skewness': float(scipy_stats.skew(self.returns)),
            'kurtosis': float(scipy_stats.kurtosis(self.returns)),
            'min':      float(np.min(self.returns)),
            'max':      float(np.max(self.returns)),
        }

    def print_moments_report(self) -> None:
        """Vypíše správu o empirických momentoch výnosov."""
        m = self.get_empirical_moments()
        print("\nEMPIRICKÉ MOMENTY VÝNOSOV:")
        print(f"  Priemer (μ)  : {m['mean']:.6f}")
        print(f"  Volatilita (σ): {m['std']:.6f}")
        print(f"  Šikmosť      : {m['skewness']:.4f}  "
              f"({'záporná - ťažký ľavý chvost' if m['skewness'] < 0 else 'kladná - ťažký pravý chvost'})")
        print(f"  Prebytočná špicatosť: {m['kurtosis']:.4f}  "
              f"({'leptokurtické - ťažké chvosty' if m['kurtosis'] > 0 else 'platykurtické - ľahké chvosty'})")
        print(f"  Normálne rozdelenie má šikmosť=0 a prebytočnú špicatosť=0")


# =============================================================================
# MONTE CARLO SIMULÁCIA
# =============================================================================

class MonteCarloSimulator:
    """
    Monte Carlo simulátor pre finančné časové rady.
    
    Používa geometrický Brownov pohyb (GBM) alebo historickú simuláciu
    na generovanie možných budúcich cenových scenárov.
    """
    
    def __init__(
        self,
        prices: np.ndarray,
        n_simulations: int = None,
        n_days: int = None,
        drift_method: str = None
    ):
        """
        Inicializácia simulátora.
        
        Args:
            prices (np.ndarray): Historické ceny
            n_simulations (int, optional): Počet simulácií
            n_days (int, optional): Počet dní na simuláciu
            drift_method (str, optional): Metóda driftu –
                'risk_neutral', 'historical' alebo 'custom'.
                Predvolene z config.SIMULATION_DRIFT_METHOD.
        """
        self.prices = np.array(prices)
        self.n_simulations = n_simulations or config.MONTE_CARLO_SIMULATIONS
        self.n_days = n_days or config.MONTE_CARLO_DAYS
        
        # Výpočet parametrov z historických dát
        self.returns = np.diff(np.log(self.prices))
        self.mu_historical = np.mean(self.returns)  # Historický priemer
        self.sigma = np.std(self.returns)            # Historická volatilita
        self.last_price = self.prices[-1]

        # Cieľový drift podľa zvolenej metódy
        self.drift_method = drift_method or config.SIMULATION_DRIFT_METHOD
        self.mu = self._resolve_drift()
        
        # Výsledky simulácie
        self.simulations = None
        self.statistics = None

    def _resolve_drift(self) -> float:
        """
        Vypočíta denný drift (μ) podľa zvolenej metódy.

        Returns:
            float: Denný log-výnosový drift
        """
        method = self.drift_method.lower()

        if method == 'historical':
            mu = self.mu_historical
            label = "historický priemer"
        elif method == 'risk_neutral':
            # Denný log-drift z ročnej bezrizikovej miery
            mu = np.log(1 + config.RISK_FREE_RATE) / 252
            label = f"rizikovo-neutrálny (rf={config.RISK_FREE_RATE:.2%})"
        elif method == 'custom':
            annual = config.SIMULATION_EXPECTED_ANNUAL_RETURN
            mu = np.log(1 + annual) / 252
            label = f"vlastný ({annual:.1%} ročne)"
        else:
            raise ValueError(
                f"Neznáma drift metóda: '{method}'. "
                "Použite 'risk_neutral', 'historical' alebo 'custom'."
            )

        print(f"[INFO] Drift metóda: {label}  →  μ_denný={mu:.6f}  "
              f"(historický μ={self.mu_historical:.6f})")
        return mu
    
    # -----------------------------------------------------------------
    # Pomocné metódy pre kontrolu extrémov
    # -----------------------------------------------------------------

    @staticmethod
    def _filter_extreme_log_returns(log_returns: np.ndarray) -> np.ndarray:
        """
        Odstránenie extrémnych denných log-výnosov resamplingom.

        Na rozdiel od orezania (clipping) táto metóda extrémne hodnoty
        neposúva na hranicu ±cap (čo by vytvorilo umelý spike v distribúcii),
        ale nahradí ich náhodným výberom z hodnôt, ktoré sú v platnom rozsahu.
        Tvar distribúcie (volatilita, šikmosť) zostáva zachovaný.
        """
        cap = config.SIMULATION_MAX_DAILY_LOG_RETURN
        extreme_mask = (log_returns < -cap) | (log_returns > cap)
        n_extreme = int(np.sum(extreme_mask))

        if n_extreme == 0:
            return log_returns

        total = log_returns.size
        print(f"  [SANITY] Nahrádzam {n_extreme} extrémnych výnosov resamplingom "
              f"({n_extreme / total * 100:.4f} %), cap=±{cap:.2f}")

        valid_values = log_returns[~extreme_mask].ravel()
        if len(valid_values) == 0:
            result = log_returns.copy()
            result[extreme_mask] = 0.0
            return result

        result = log_returns.copy()
        idx = np.random.randint(0, len(valid_values), size=n_extreme)
        result[extreme_mask] = valid_values[idx]
        return result

    def _apply_price_floor(self) -> None:
        """Zabezpečí, že žiadna simulovaná cena neklesne pod minimálnu hranicu."""
        floor = config.SIMULATION_MIN_PRICE
        below = np.sum(self.simulations < floor)
        if below > 0:
            self.simulations = np.maximum(self.simulations, floor)
            print(f"  [SANITY] {below} cenových bodov zvýšených na minimálnu cenu {floor}")

    def sanity_check(self) -> Dict[str, Any]:
        """
        Vykoná kontrolu realizmu simulovaných výsledkov.

        Skontroluje priemerný ročný výnos, extrémne cenové pomery
        a podiel ciest, ktoré sa správajú nerealisticky.

        Returns:
            Dict: Výsledky kontroly s varovaniami (ak existujú)
        """
        if self.simulations is None:
            raise ValueError("Najprv spustite simuláciu.")

        final_prices = self.simulations[:, -1]
        price_ratio = final_prices / self.last_price
        years = self.n_days / 252
        annual_returns = (price_ratio ** (1 / years) - 1) * 100 if years > 0 else (price_ratio - 1) * 100

        mean_annual = float(np.mean(annual_returns))
        median_annual = float(np.median(annual_returns))
        pct_extreme_high = float((price_ratio > config.SANITY_MAX_PRICE_RATIO).mean() * 100)
        pct_near_zero = float((final_prices < 1.0).mean() * 100)

        warnings_list: List[str] = []

        if mean_annual > config.SANITY_MAX_ANNUAL_RETURN_PCT:
            warnings_list.append(
                f"Priemerný ročný výnos ({mean_annual:.1f} %) prekračuje realistický prah "
                f"({config.SANITY_MAX_ANNUAL_RETURN_PCT} %).")
        if mean_annual < config.SANITY_MIN_ANNUAL_RETURN_PCT:
            warnings_list.append(
                f"Priemerný ročný výnos ({mean_annual:.1f} %) je pod realistickým prahom "
                f"({config.SANITY_MIN_ANNUAL_RETURN_PCT} %).")
        if pct_extreme_high > 1.0:
            warnings_list.append(
                f"{pct_extreme_high:.2f} % ciest prekračuje {config.SANITY_MAX_PRICE_RATIO}× "
                f"počiatočnej ceny.")
        if pct_near_zero > 5.0:
            warnings_list.append(
                f"{pct_near_zero:.2f} % ciest kleslo pod 1 USD.")

        result = {
            'mean_annual_return_pct': mean_annual,
            'median_annual_return_pct': median_annual,
            'pct_extreme_high': pct_extreme_high,
            'pct_near_zero': pct_near_zero,
            'n_warnings': len(warnings_list),
            'warnings': warnings_list,
            'is_realistic': len(warnings_list) == 0,
        }

        # Výpis
        print(f"\n{'='*60}")
        print("SANITY CHECK SIMULÁCIE")
        print(f"{'='*60}")
        print(f"  Priemerný ročný výnos: {mean_annual:.2f} %")
        print(f"  Medián ročného výnosu: {median_annual:.2f} %")
        print(f"  Extrémne vysoké cesty (>{config.SANITY_MAX_PRICE_RATIO}×): {pct_extreme_high:.2f} %")
        print(f"  Cesty blízko nuly (<1 USD): {pct_near_zero:.2f} %")
        if warnings_list:
            print(f"\n  ⚠ VAROVANIA ({len(warnings_list)}):")
            for w in warnings_list:
                print(f"    - {w}")
        else:
            print(f"\n  ✓ Simulácia prešla kontrolou realizmu.")
        print(f"{'='*60}")

        return result

    def simulate_gbm(self, seed: int = None) -> np.ndarray:
        """
        Vykoná Monte Carlo simuláciu pomocou geometrického Brownovho pohybu.
        
        GBM predpokladá, že logaritmické výnosy sú normálne rozdelené.
        
        dS = μS dt + σS dW
        
        Args:
            seed (int, optional): Random seed pre reprodukovateľnosť
            
        Returns:
            np.ndarray: Matica simulovaných cien (n_simulations x n_days)
        """
        print(f"[INFO] Spúšťam Monte Carlo simuláciu (GBM)...")
        print(f"  - Počet simulácií: {self.n_simulations}")
        print(f"  - Počet dní: {self.n_days}")
        print(f"  - Počiatočná cena: {self.last_price:.2f}")
        print(f"  - Drift metóda: {self.drift_method}")
        print(f"  - Použitý drift (μ): {self.mu:.6f}  (historický: {self.mu_historical:.6f})")
        print(f"  - Denná volatilita (σ): {self.sigma:.6f}")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Drift a diffusion
        dt = 1  # Denný krok
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        # Generovanie náhodných šokov
        random_shocks = np.random.standard_normal((self.n_simulations, self.n_days))
        
        # Výpočet log výnosov
        log_returns = drift + diffusion * random_shocks
        log_returns = self._filter_extreme_log_returns(log_returns)
        
        # Kumulatívne log výnosy
        cumulative_log_returns = np.cumsum(log_returns, axis=1)
        
        # Konverzia na ceny
        self.simulations = self.last_price * np.exp(cumulative_log_returns)
        
        # Pridanie počiatočnej ceny
        initial_prices = np.full((self.n_simulations, 1), self.last_price)
        self.simulations = np.hstack([initial_prices, self.simulations])
        self._apply_price_floor()
        
        print(f"[OK] GBM simulácia dokončená")

        return self.simulations

    def simulate_with_fitted_distribution(
        self,
        fitter: ReturnDistributionFitter,
        seed: int = None
    ) -> np.ndarray:
        """
        Vykoná Monte Carlo simuláciu s empiricky nafitovaným rozdelením výnosov.

        Toto je jadro hybridného prístupu: namiesto predpokladu normálneho
        rozdelenia (štandardný GBM) sa použije rozdelenie nafitované priamo
        na historické log-výnosy daného tickera.

        Rovnica cenovej cesty:
            S(t+1) = S(t) * exp(r_t)
        kde r_t ~ nafitované rozdelenie (napr. Student-t, Johnson SU, ...)

        Args:
            fitter (ReturnDistributionFitter): Nafitovaný distribučný fitter
            seed (int, optional): Radnom seed

        Returns:
            np.ndarray: Matica simulovaných cien (n_simulations x (n_days+1))
        """
        if not fitter.fit_complete:
            raise ValueError("ReturnDistributionFitter musí mať dokončený fit(). "
                             "Zavolajte fitter.fit() pred simuláciou.")

        dist_label = fitter.all_results[fitter.best_distribution]['name_sk']
        print(f"[INFO] Spúšťam MC simuláciu s nafitovaným rozdelením...")
        print(f"  - Rozdelenie     : {dist_label} ({fitter.best_distribution})")
        print(f"  - Počet simulácií: {self.n_simulations}")
        print(f"  - Počet dní      : {self.n_days}")
        print(f"  - Počiatočná cena: {self.last_price:.2f}")

        if seed is not None:
            np.random.seed(seed)

        # Vygeneruj log-výnosy z nafitovanej distribúcie
        sampled = fitter.generate_samples(
            self.n_simulations * self.n_days, seed=seed
        ).reshape(self.n_simulations, self.n_days)

        # Precentrovanie na cieľový drift (zachová tvar rozdelenia –
        # volatilitu, šikmosť, špicatosť – ale posunie priemer)
        sampled = sampled - np.mean(sampled) + self.mu

        # Nahradenie extrémov resamplingom
        sampled = self._filter_extreme_log_returns(sampled)

        # Kumulatívne log-výnosy → ceny
        cum_log_returns = np.cumsum(sampled, axis=1)
        self.simulations = self.last_price * np.exp(cum_log_returns)

        # Prepend počiatočnú cenu
        initial_col = np.full((self.n_simulations, 1), self.last_price)
        self.simulations = np.hstack([initial_col, self.simulations])
        self._apply_price_floor()

        print(f"[OK] Simulácia s nafitovaným rozdelením dokončená")

        return self.simulations

    def simulate_historical(self, seed: int = None) -> np.ndarray:
        """
        Vykoná Monte Carlo simuláciu pomocou historickej metódy.
        
        Náhodne vyberá skutočné historické výnosy (bootstrap).
        
        Args:
            seed (int, optional): Random seed
            
        Returns:
            np.ndarray: Matica simulovaných cien
        """
        print(f"[INFO] Spúšťam Monte Carlo simuláciu (historická metóda)...")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Bootstrap - náhodný výber z historických výnosov
        sampled_returns = np.random.choice(
            self.returns, 
            size=(self.n_simulations, self.n_days),
            replace=True
        )
        
        # Precentrovanie na cieľový drift
        sampled_returns = sampled_returns - np.mean(sampled_returns) + self.mu

        # Nahradenie extrémov resamplingom
        sampled_returns = self._filter_extreme_log_returns(sampled_returns)

        # Kumulatívne výnosy
        cumulative_returns = np.cumsum(sampled_returns, axis=1)
        
        # Konverzia na ceny
        self.simulations = self.last_price * np.exp(cumulative_returns)
        
        # Pridanie počiatočnej ceny
        initial_prices = np.full((self.n_simulations, 1), self.last_price)
        self.simulations = np.hstack([initial_prices, self.simulations])
        self._apply_price_floor()
        
        print(f"[OK] Historická bootstrap simulácia dokončená")

        return self.simulations

    def calculate_statistics(self) -> Dict[str, float]:
        """
        Vypočíta štatistiky zo simulácií.
        
        Returns:
            Dict[str, float]: Slovník so štatistikami
        """
        if self.simulations is None:
            raise ValueError("Najprv spustite simuláciu.")
        
        final_prices = self.simulations[:, -1]
        
        self.statistics = {
            'pociatocna_cena': self.last_price,
            'priemerna_konecna_cena': np.mean(final_prices),
            'medianová_konecna_cena': np.median(final_prices),
            'std_konecna_cena': np.std(final_prices),
            'min_konecna_cena': np.min(final_prices),
            'max_konecna_cena': np.max(final_prices),
            'percentil_5': np.percentile(final_prices, 5),
            'percentil_25': np.percentile(final_prices, 25),
            'percentil_75': np.percentile(final_prices, 75),
            'percentil_95': np.percentile(final_prices, 95),
            'priemerny_vynos': (np.mean(final_prices) / self.last_price - 1) * 100,
            'pravdepodobnost_rastu': (final_prices > self.last_price).mean() * 100,
        }
        
        return self.statistics
    
    def calculate_var(
        self,
        confidence_level: float = 0.95,
        horizon: int = None
    ) -> Dict[str, float]:
        """
        Vypočíta Value at Risk (VaR) a Conditional VaR (CVaR/ES).
        
        VaR je maximálna očakávaná strata pri danej úrovni spoľahlivosti.
        CVaR je priemerná strata pri prekročení VaR.
        
        Args:
            confidence_level (float): Úroveň spoľahlivosti (napr. 0.95)
            horizon (int, optional): Horizont v dňoch (predvolené: koniec simulácie)
            
        Returns:
            Dict[str, float]: VaR a CVaR hodnoty
        """
        if self.simulations is None:
            raise ValueError("Najprv spustite simuláciu.")
        
        if horizon is None:
            horizon = self.n_days
        
        # Výnosy k danému horizontu
        prices_at_horizon = self.simulations[:, horizon]
        returns = (prices_at_horizon - self.last_price) / self.last_price * 100
        
        # VaR (percentil strát)
        var_percentile = (1 - confidence_level) * 100
        var = -np.percentile(returns, var_percentile)
        
        # CVaR (Expected Shortfall)
        losses = returns[returns <= -var]
        cvar = -np.mean(losses) if len(losses) > 0 else var
        
        result = {
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level,
            'horizon_days': horizon,
            'interpretation_var': f"S {confidence_level*100:.0f}% pravdepodobnosťou nestratíme viac ako {var:.2f}%",
            'interpretation_cvar': f"Priemerná strata pri prekročení VaR je {cvar:.2f}%"
        }
        
        return result
    
    def print_report(self) -> None:
        """Vypíše súhrnný report simulácie."""
        if self.simulations is None:
            print("[CHYBA] Žiadna simulácia nebola vykonaná.")
            return
        
        stats = self.calculate_statistics()
        var_95 = self.calculate_var(0.95)
        var_99 = self.calculate_var(0.99)
        
        print("\n" + "=" * 60)
        print("MONTE CARLO SIMULÁCIA - SÚHRNNÝ REPORT")
        print("=" * 60)
        
        print(f"\nParametre simulácie:")
        print(f"  Počet simulácií: {self.n_simulations}")
        print(f"  Horizont: {self.n_days} dní")
        print(f"  Počiatočná cena: {stats['pociatocna_cena']:.2f}")
        
        print(f"\nŠtatistiky koncových cien:")
        print(f"  Priemerná: {stats['priemerna_konecna_cena']:.2f}")
        print(f"  Medián: {stats['medianová_konecna_cena']:.2f}")
        print(f"  Štd. odchýlka: {stats['std_konecna_cena']:.2f}")
        print(f"  Minimum: {stats['min_konecna_cena']:.2f}")
        print(f"  Maximum: {stats['max_konecna_cena']:.2f}")
        
        print(f"\nPercentily:")
        print(f"  5. percentil: {stats['percentil_5']:.2f}")
        print(f"  25. percentil: {stats['percentil_25']:.2f}")
        print(f"  75. percentil: {stats['percentil_75']:.2f}")
        print(f"  95. percentil: {stats['percentil_95']:.2f}")
        
        print(f"\nOčakávaný výnos: {stats['priemerny_vynos']:.2f}%")
        print(f"Pravdepodobnosť rastu: {stats['pravdepodobnost_rastu']:.1f}%")
        
        print(f"\nRizikové metriky:")
        print(f"  VaR (95%): {var_95['var']:.2f}%")
        print(f"  CVaR (95%): {var_95['cvar']:.2f}%")
        print(f"  VaR (99%): {var_99['var']:.2f}%")
        print(f"  CVaR (99%): {var_99['cvar']:.2f}%")
        
        print("=" * 60)


# =============================================================================
# BACKTESTING
# =============================================================================

class Backtester:
    """
    Backtester pre vyhodnotenie obchodných stratégií.
    
    Simuluje obchodovanie na historických dátach a vyhodnocuje
    výkonnosť stratégie založenej na predikciách modelu.
    """
    
    def __init__(
        self,
        initial_capital: float = None,
        transaction_cost: float = None,
        risk_free_rate: float = None
    ):
        """
        Inicializácia backtestera.
        
        Args:
            initial_capital (float, optional): Počiatočný kapitál
            transaction_cost (float, optional): Transakčné náklady (pomer)
            risk_free_rate (float, optional): Bezriziková úroková miera
        """
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.transaction_cost = transaction_cost or config.TRANSACTION_COST
        self.risk_free_rate = risk_free_rate or config.RISK_FREE_RATE
        
        # Výsledky backtestingu
        self.results = None
        self.trades = []
        self.equity_curve = []
        self.positions = []
    
    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        dates: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Spustí backtest stratégie.
        
        Args:
            prices (np.ndarray): Historické ceny
            signals (np.ndarray): Obchodné signály (1=kúpiť/držať, 0=predať/čakať)
            dates (pd.Series, optional): Dátumy pre záznamy
            
        Returns:
            Dict[str, Any]: Výsledky backtestingu
        """
        print("[INFO] Spúšťam backtest stratégie...")
        
        prices = np.array(prices)
        signals = np.array(signals)
        
        if len(prices) != len(signals):
            raise ValueError("Dĺžka cien a signálov sa nezhoduje.")
        
        n = len(prices)
        
        # Inicializácia
        capital = self.initial_capital
        position = 0  # Počet akcií
        self.equity_curve = [capital]
        self.trades = []
        self.positions = [0]
        
        # Simulácia obchodovania
        for i in range(1, n):
            current_price = prices[i]
            previous_price = prices[i-1]
            signal = signals[i]
            
            # Aktuálna hodnota portfólia
            portfolio_value = capital + position * current_price
            
            # Zmena pozície podľa signálu
            if signal == 1 and position == 0:
                # Kúpa - investujeme všetok kapitál
                shares_to_buy = capital * (1 - self.transaction_cost) / current_price
                position = shares_to_buy
                capital = 0
                
                self.trades.append({
                    'den': i,
                    'datum': dates.iloc[i] if dates is not None else i,
                    'typ': 'KÚPA',
                    'cena': current_price,
                    'pocet_akcii': shares_to_buy,
                    'hodnota': shares_to_buy * current_price
                })
                
            elif signal == 0 and position > 0:
                # Predaj - predáme všetko
                sale_value = position * current_price * (1 - self.transaction_cost)
                
                self.trades.append({
                    'den': i,
                    'datum': dates.iloc[i] if dates is not None else i,
                    'typ': 'PREDAJ',
                    'cena': current_price,
                    'pocet_akcii': position,
                    'hodnota': sale_value
                })
                
                capital = sale_value
                position = 0
            
            # Aktualizácia equity curve
            portfolio_value = capital + position * current_price
            self.equity_curve.append(portfolio_value)
            self.positions.append(position)
        
        # Záverečný predaj ak ešte držíme pozíciu na konci obdobia
        if position > 0:
            final_value = position * prices[-1] * (1 - self.transaction_cost)
            self.trades.append({
                'den': n - 1,
                'datum': dates.iloc[n - 1] if dates is not None else n - 1,
                'typ': 'PREDAJ (záverečný)',
                'cena': prices[-1],
                'pocet_akcii': position,
                'hodnota': final_value
            })
            # Aktualizuj poslednú hodnotu equity curve o transakčné náklady
            self.equity_curve[-1] = final_value
            position = 0

        # Výpočet metrík
        self.results = self._calculate_metrics(prices, signals, dates)

        print(f"[OK] Backtest dokončený. Finálna hodnota portfólia: {self.equity_curve[-1]:.2f}")
        
        return self.results
    
    def _calculate_metrics(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        dates: pd.Series
    ) -> Dict[str, Any]:
        """
        Vypočíta metriky výkonnosti stratégie.
        
        Args:
            prices (np.ndarray): Ceny
            signals (np.ndarray): Signály
            dates (pd.Series): Dátumy
            
        Returns:
            Dict[str, Any]: Metriky
        """
        equity = np.array(self.equity_curve)
        
        # Základné metriky
        final_value = equity[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Denné výnosy
        daily_returns = np.diff(equity) / equity[:-1] * 100
        
        # Ročné metriky (predpoklad 252 obchodných dní)
        n_days = len(equity)
        years = n_days / 252
        annual_return = ((final_value / self.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Volatilita
        daily_volatility = np.std(daily_returns)
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # Sharpe Ratio
        excess_return = annual_return - self.risk_free_rate * 100
        sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        max_drawdown = np.min(drawdown)
        
        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino Ratio (používa len negatívne výnosy)
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        # Štatistiky obchodov
        n_trades = len(self.trades)
        n_buys = sum(1 for t in self.trades if t['typ'] == 'KÚPA')
        n_sells = sum(1 for t in self.trades if t['typ'] == 'PREDAJ')
        
        # Buy & Hold porovnanie
        buy_hold_return = (prices[-1] / prices[0] - 1) * 100
        
        # Win rate z obchodov
        trade_returns = []
        for i in range(0, len(self.trades) - 1, 2):
            if i + 1 < len(self.trades):
                buy_price = self.trades[i]['cena']
                sell_price = self.trades[i + 1]['cena']
                trade_return = (sell_price / buy_price - 1) * 100
                trade_returns.append(trade_return)
        
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100 if trade_returns else 0
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        
        metrics = {
            'pociatocny_kapital': self.initial_capital,
            'konecna_hodnota': final_value,
            'celkovy_vynos_pct': total_return,
            'rocny_vynos_pct': annual_return,
            'rocna_volatilita_pct': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown_pct': max_drawdown,
            'pocet_obchodov': n_trades,
            'pocet_kupnych_signalov': n_buys,
            'pocet_predajnych_signalov': n_sells,
            'win_rate_pct': win_rate,
            'priemerny_vynos_obchodu_pct': avg_trade_return,
            'buy_hold_vynos_pct': buy_hold_return,
            'prekonanie_buy_hold_pct': total_return - buy_hold_return,
            'equity_curve': equity.tolist(),
            'trade_returns': trade_returns,
            'cumulative_returns': ((equity / self.initial_capital) - 1) * 100,
        }
        
        return metrics
    
    def print_report(self) -> None:
        """Vypíše report backtestingu."""
        if self.results is None:
            print("[CHYBA] Žiadny backtest nebol vykonaný.")
            return
        
        r = self.results
        
        print("\n" + "=" * 60)
        print("BACKTEST REPORT")
        print("=" * 60)
        
        print(f"\nKapitál:")
        print(f"  Počiatočný: {r['pociatocny_kapital']:,.2f} USD")
        print(f"  Konečný: {r['konecna_hodnota']:,.2f} USD")
        print(f"  Celkový výnos: {r['celkovy_vynos_pct']:.2f}%")
        
        print(f"\nVýkonnosť:")
        print(f"  Ročný výnos: {r['rocny_vynos_pct']:.2f}%")
        print(f"  Ročná volatilita: {r['rocna_volatilita_pct']:.2f}%")
        
        print(f"\nRizikovo-upravené metriky:")
        print(f"  Sharpe Ratio: {r['sharpe_ratio']:.4f}")
        print(f"  Sortino Ratio: {r['sortino_ratio']:.4f}")
        print(f"  Calmar Ratio: {r['calmar_ratio']:.4f}")
        print(f"  Maximum Drawdown: {r['max_drawdown_pct']:.2f}%")
        
        print(f"\nObchody:")
        print(f"  Počet obchodov: {r['pocet_obchodov']}")
        print(f"  Win Rate: {r['win_rate_pct']:.1f}%")
        print(f"  Priemerný výnos/obchod: {r['priemerny_vynos_obchodu_pct']:.2f}%")
        
        print(f"\nPorovnanie s Buy & Hold:")
        print(f"  Buy & Hold výnos: {r['buy_hold_vynos_pct']:.2f}%")
        print(f"  Prekonanie B&H: {r['prekonanie_buy_hold_pct']:.2f}%")
        
        verdict = "LEPŠIA" if r['prekonanie_buy_hold_pct'] > 0 else "HORŠIA"
        print(f"\n  -> Stratégia je {verdict} ako Buy & Hold")
        
        print("=" * 60)
    
    def get_trades_df(self) -> pd.DataFrame:
        """
        Vráti DataFrame so všetkými obchodmi.
        
        Returns:
            pd.DataFrame: Tabuľka obchodov
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def get_results_for_visualization(self) -> Dict[str, Any]:
        """
        Pripraví výsledky pre vizualizáciu.
        
        Returns:
            Dict: Dáta pre vizualizáciu
        """
        if self.results is None:
            return {}
        
        return {
            'equity_curve': np.array(self.equity_curve),
            'cumulative_returns': self.results['cumulative_returns'],
            'trade_returns': self.results['trade_returns'],
            'metrics': {
                'Celkový výnos': f"{self.results['celkovy_vynos_pct']:.2f}%",
                'Ročný výnos': f"{self.results['rocny_vynos_pct']:.2f}%",
                'Sharpe Ratio': f"{self.results['sharpe_ratio']:.4f}",
                'Max Drawdown': f"{self.results['max_drawdown_pct']:.2f}%",
                'Win Rate': f"{self.results['win_rate_pct']:.1f}%",
                'Počet obchodov': self.results['pocet_obchodov'],
            }
        }


# =============================================================================
# STRATÉGIE
# =============================================================================

def create_model_based_strategy(
    model,
    X: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Vytvorí obchodné signály z predikcií modelu.
    
    Args:
        model: Natrénovaný model s metódou predict
        X (np.ndarray): Features pre predikciu
        threshold (float): Prah pre signál (pre klasifikáciu)
        
    Returns:
        np.ndarray: Obchodné signály (1=kúpiť, 0=predať)
    """
    # Predikcia
    if hasattr(model, 'predict_proba'):
        # Klasifikačný model
        try:
            probas = model.predict_proba(X)
            if len(probas.shape) > 1:
                probas = probas[:, 1]
            signals = (probas >= threshold).astype(int)
        except Exception:
            predictions = model.predict(X)
            signals = (predictions >= threshold).astype(int)
    else:
        predictions = model.predict(X)
        # Pre regresiu: signál ak predikujeme rast
        signals = (predictions > 0).astype(int)
    
    return signals


def moving_average_crossover_strategy(
    prices: np.ndarray,
    short_window: int = 20,
    long_window: int = 50
) -> np.ndarray:
    """
    Vytvorí signály podľa kríženia kĺzavých priemerov.
    
    Kúpny signál: krátky MA prekríži dlhý MA zdola
    Predajný signál: krátky MA prekríži dlhý MA zhora
    
    Args:
        prices (np.ndarray): Ceny
        short_window (int): Okno pre krátky MA
        long_window (int): Okno pre dlhý MA
        
    Returns:
        np.ndarray: Obchodné signály
    """
    prices = np.array(prices)
    n = len(prices)
    
    # Výpočet MA
    short_ma = np.zeros(n)
    long_ma = np.zeros(n)
    
    for i in range(n):
        if i >= short_window - 1:
            short_ma[i] = np.mean(prices[i - short_window + 1:i + 1])
        if i >= long_window - 1:
            long_ma[i] = np.mean(prices[i - long_window + 1:i + 1])
    
    # Signály
    signals = np.zeros(n)
    for i in range(long_window, n):
        if short_ma[i] > long_ma[i]:
            signals[i] = 1  # Kúpiť/držať
        else:
            signals[i] = 0  # Predať/čakať
    
    return signals


def rsi_strategy(
    prices: np.ndarray,
    period: int = 14,
    oversold: int = 30,
    overbought: int = 70
) -> np.ndarray:
    """
    Vytvorí signály podľa RSI.
    
    Kúpny signál: RSI < oversold
    Predajný signál: RSI > overbought
    
    Args:
        prices (np.ndarray): Ceny
        period (int): Perióda RSI
        oversold (int): Hranica prepredanosti
        overbought (int): Hranica prekúpenosti
        
    Returns:
        np.ndarray: Obchodné signály
    """
    prices = np.array(prices)
    n = len(prices)
    
    # Výpočet RSI
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n - 1)
    avg_loss = np.zeros(n - 1)
    
    # Prvý priemer
    avg_gain[period - 1] = np.mean(gain[:period])
    avg_loss[period - 1] = np.mean(loss[:period])
    
    # EMA
    for i in range(period, n - 1):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    # RSI
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Signály
    signals = np.zeros(n)
    position = 0
    
    for i in range(period, n):
        current_rsi = rsi[i - 1] if i > 0 else 50
        
        if current_rsi < oversold and position == 0:
            signals[i] = 1
            position = 1
        elif current_rsi > overbought and position == 1:
            signals[i] = 0
            position = 0
        else:
            signals[i] = position
    
    return signals


# =============================================================================
# ANALÝZA CITLIVOSTI
# =============================================================================

def sensitivity_analysis(
    backtester_func: Callable,
    param_name: str,
    param_values: List[Any],
    base_params: Dict
) -> pd.DataFrame:
    """
    Vykoná analýzu citlivosti na parameter stratégie.
    
    Args:
        backtester_func (Callable): Funkcia pre backtest
        param_name (str): Názov parametra
        param_values (List): Hodnoty parametra na testovanie
        base_params (Dict): Základné parametre
        
    Returns:
        pd.DataFrame: Výsledky pre rôzne hodnoty parametra
    """
    print(f"[INFO] Analýza citlivosti pre parameter '{param_name}'...")
    
    results = []
    
    for value in param_values:
        params = base_params.copy()
        params[param_name] = value
        
        try:
            result = backtester_func(**params)
            results.append({
                param_name: value,
                'celkovy_vynos': result.get('celkovy_vynos_pct', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'max_drawdown': result.get('max_drawdown_pct', 0),
            })
        except Exception as e:
            print(f"  [VAROVANIE] Chyba pre {param_name}={value}: {e}")
    
    df = pd.DataFrame(results)
    print(f"[OK] Analýza citlivosti dokončená")
    
    return df


# =============================================================================
# HLAVNÁ FUNKCIA PRE TESTOVANIE MODULU
# =============================================================================

if __name__ == "__main__":
    import data_downloader
    
    print("=" * 60)
    print("TESTOVANIE MODULU SIMULATION")
    print("=" * 60)
    
    # Načítanie dát
    try:
        df = data_downloader.load_stock_data("AAPL")
    except FileNotFoundError:
        df = data_downloader.download_stock_data("AAPL")
    
    prices = df['Close'].values
    
    # Test Monte Carlo simulácie
    print("\n--- Test Monte Carlo simulácie ---")
    mc = MonteCarloSimulator(prices, n_simulations=1000, n_days=252)
    simulations = mc.simulate_gbm(seed=42)
    mc.print_report()
    
    # Test VaR
    var_result = mc.calculate_var(0.95)
    print(f"\n{var_result['interpretation_var']}")
    print(f"{var_result['interpretation_cvar']}")
    
    # Test backtestingu
    print("\n--- Test backtestingu ---")
    
    # MA crossover stratégia
    signals = moving_average_crossover_strategy(prices, short_window=20, long_window=50)
    
    bt = Backtester()
    results = bt.run(prices, signals)
    bt.print_report()
    
    # Test RSI stratégie
    print("\n--- Test RSI stratégie ---")
    rsi_signals = rsi_strategy(prices)
    
    bt_rsi = Backtester()
    results_rsi = bt_rsi.run(prices, rsi_signals)
    bt_rsi.print_report()
    
    print("\n[OK] Testovanie modulu dokončené")
