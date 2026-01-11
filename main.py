# portfolio_bl_mbte_yf.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
import yfinance as yf
from datetime import datetime

# -----------------------------
# Ticker sets (from your brief)
# -----------------------------
SP500_TICKERS = [
    'AAPL','MSFT','GOOGL','AMZN','TSLA','META',"NVDA",'BRK-B','UNH','JNJ','JPM','V',
    'PG','XOM','HD','CVX','MA','BAC','ABBV','PFE','AVGO','COST','DIS','KO','MRK',
    'TMO','PEP','WMT','ABT','NFLX','ADBE','CRM','ACN','VZ','CMCSA','DHR','NKE',
    'TXN','BMY','NEE','QCOM','PM','RTX','UPS','LOW','T','HON','AMGN','IBM','CAT'
]

SECTOR_FACTORS = ['XLF','XLE','XLV',"XLI","XLK"]  # Financials, Energy, Industrials, Tech, Healthcare
STYLE_FACTORS  = ['VMOT','VB','QUAL','MTUM']      # Value, Size, Quality, Momentum

BENCHMARKS = {
    'SPY': 'S&P 500 (Cap-Weighted)',
    'RSP': 'S&P 500 (Equal-Weighted)',
    'IVE': 'S&P 500 Value',
    'IVW': 'S&P 500 Growth',
    'QQQ': 'NASDAQ 100',
    "TLT" : "TLT",
    "GLD":"GLD",
    "HYG": "HYG",
    "LQD": "LQD"
}

# -----------------------------
# Data utilities
# -----------------------------
def _dl_adj_close(tickers, start, end):
    """Download Adj Close panel for tickers and return a clean wide DataFrame."""
    if isinstance(tickers, (list, tuple, set)):
        tickers = list(tickers)
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    # yfinance returns columns either as single-level (when 1 ticker) or multi-level
    if isinstance(data.columns, pd.MultiIndex):
        adj = data['Close'].copy()
    else:
        adj = data[['Close']].rename(columns={'Close': tickers if isinstance(tickers, str) else tickers[0]})
    # Some tickers might not pull; drop all-null cols
    adj = adj.dropna(axis=1, how='all')
    return adj

def _pct_change(df):
    return df.sort_index().pct_change().dropna(how='all')

def load_market_data(start='2015-01-01', end=None):
    """
    Returns:
      assets_ret  : DataFrame (daily % returns) for 50 S&P tickers
      sector_ret  : DataFrame for sector ETFs
      style_ret   : DataFrame for style ETFs
      bench_ret   : DataFrame for benchmark ETFs
      rfr_annual  : float annualized risk-free (from ^IRX), e.g., 0.04
    """
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    # Assets
    prices_assets = _dl_adj_close(SP500_TICKERS, start, end)
    assets_ret = _pct_change(prices_assets).dropna(axis=1, how='any')  # keep only tickers with data

    # Sector + Style factors
    prices_sector = _dl_adj_close(SECTOR_FACTORS, start, end)
    sector_ret = _pct_change(prices_sector)

    prices_style = _dl_adj_close(STYLE_FACTORS, start, end)
    style_ret = _pct_change(prices_style)

    # Benchmarks
    prices_bench = _dl_adj_close(list(BENCHMARKS.keys()), start, end)
    bench_ret = _pct_change(prices_bench)

    # Align all on common dates
    common_idx = assets_ret.index
    for df in [sector_ret, style_ret, bench_ret]:
        common_idx = common_idx.intersection(df.index)

    assets_ret = assets_ret.loc[common_idx]
    sector_ret = sector_ret.loc[common_idx]
    style_ret  = style_ret.loc[common_idx]
    bench_ret  = bench_ret.loc[common_idx]

    # Risk-free from 13-week T-bill (^IRX) — Yahoo gives % yield. Convert to decimal.
    irx = yf.download('^IRX', start=start, end=end, progress=False)['Close'].dropna()
    # Smooth a bit to avoid a single-day blip; use last 60 trading days average
    if len(irx) >= 60:
        rfr_annual = float(irx.iloc[-60:].mean() / 100.0)
    else:
        rfr_annual = float(irx.iloc[-1] / 100.0)

    # Basic hygiene: remove any columns with too many NaNs (rare with these ETFs)
    min_valid = int(0.9 * len(common_idx))
    assets_ret = assets_ret.loc[:, assets_ret.count() >= min_valid]

    return assets_ret, sector_ret, style_ret, bench_ret, rfr_annual

# -----------------------------
# Optimizer classes (your originals, kept)
# -----------------------------
class PortfolioOptimizer:
    def __init__(self, returns_data, risk_free_rate=0.02):
        self.returns = returns_data
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns_data.columns)
        self.asset_names = returns_data.columns.tolist()

        # Annualized stats
        self.mean_returns = returns_data.mean() * 252
        self.cov_matrix   = returns_data.cov() * 252
        self.corr_matrix  = returns_data.corr()

        # Higher moments (daily)
        self.skewness = returns_data.apply(skew)
        self.excess_kurtosis = returns_data.apply(lambda x: kurtosis(x, fisher=True))

        print(f"Initialized optimizer with {self.n_assets} assets")
        print(f"Sample period: {len(returns_data)} observations ({returns_data.index.min().date()} → {returns_data.index.max().date()})")

class BlackLittermanOptimizer(PortfolioOptimizer):
    def __init__(self, returns_data, market_caps=None, risk_aversion=3.0, tau=0.05, risk_free_rate=0.02):
        super().__init__(returns_data, risk_free_rate)
        self.risk_aversion = risk_aversion
        self.tau = tau

        if market_caps is None:
            self.market_weights = np.ones(self.n_assets) / self.n_assets
        else:
            self.market_weights = np.array(market_caps) / np.sum(market_caps)

        self.equilibrium_returns = self._calculate_equilibrium_returns()

    def _calculate_equilibrium_returns(self):
        return self.risk_aversion * np.dot(self.cov_matrix, self.market_weights)

    def update_views(self, P, Q, omega):
        self.P, self.Q, self.omega = P, Q, omega
        tau_cov = self.tau * self.cov_matrix

        M1 = np.linalg.inv(tau_cov)
        M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
        M3 = np.dot(np.linalg.inv(tau_cov), self.equilibrium_returns)
        M4 = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))

        self.bl_returns = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
        # NOTE: Many implementations use the market covariance for optimization.
        # We keep your original line but you may prefer: self.bl_cov = self.cov_matrix
        self.bl_cov = np.linalg.inv(M1 + M2)

        return self.bl_returns, self.bl_cov

class MBTEOptimizer(PortfolioOptimizer):
    def __init__(self, returns_data, benchmark_returns, risk_free_rate=0.02):
        super().__init__(returns_data, risk_free_rate)
        self.benchmarks = benchmark_returns if isinstance(benchmark_returns, pd.DataFrame) else benchmark_returns.to_frame()
        self.n_benchmarks = len(self.benchmarks.columns)
        self.benchmark_names = self.benchmarks.columns.tolist()
        self.benchmark_returns = self.benchmarks.mean() * 252
        self.benchmark_cov = self.benchmarks.cov() * 252
        print(f"MBTE initialized with {self.n_benchmarks} benchmarks: {', '.join(self.benchmark_names)}")

# -----------------------------
# Metrics & helpers
# -----------------------------
def portfolio_performance(weights, returns, cov_matrix, risk_free_rate=0.02):
    port_ret = float(np.dot(weights, returns))
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else np.nan
    return port_ret, port_vol, sharpe

def tracking_error(weights, asset_returns_df, benchmark_returns):
    """
    Annualized TE. If benchmark_returns is DataFrame -> TE averaged across all benchmarks.
    """
    port_series = asset_returns_df.dot(weights)
    if isinstance(benchmark_returns, pd.DataFrame):
        tes = []
        for col in benchmark_returns.columns:
            te = (port_series - benchmark_returns[col]).std() * np.sqrt(252)
            tes.append(te)
        return float(np.mean(tes))
    else:
        return float((port_series - benchmark_returns).std() * np.sqrt(252))

# -----------------------------
# Analyses (now using real data)
# -----------------------------
def mean_variance_analysis(returns_data, benchmark_data, risk_free_rate):
    print("\n" + "="*60)
    print("PORTFOLIO A: MEAN-VARIANCE ANALYSIS (Real yfinance data)")
    print("="*60)

    asset_names = list(returns_data.columns)
    n_assets = len(asset_names)

    base_opt = PortfolioOptimizer(returns_data, risk_free_rate)
    bl_opt   = BlackLittermanOptimizer(returns_data, risk_free_rate=risk_free_rate)
    mbte_opt = MBTEOptimizer(returns_data, benchmark_data, risk_free_rate=risk_free_rate)

    # Example BL views (tweak as desired): overweight first asset, underweight a bond-proxy if present
    # Here we simply place views on two liquid mega-caps if available
    candidates = [t for t in ['AAPL','MSFT','JPM','XOM'] if t in asset_names]
    if len(candidates) >= 2:
        idx_a = asset_names.index(candidates[0])
        idx_b = asset_names.index(candidates[1])
    else:
        idx_a, idx_b = 0, 1

    P = np.zeros((2, n_assets))
    P[0, idx_a] = 1.0
    P[1, idx_b] = 1.0
    Q = np.array([0.02, -0.01])     # annualized excess returns view
    omega = np.diag([0.001, 0.001]) # view uncertainty
    bl_opt.update_views(P, Q, omega)

    def optimize_portfolio(expected_returns, cov_matrix, objective='min_vol', target_return=None):
        def obj(w):
            if objective == 'min_vol':
                return float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))
            elif objective == 'max_sharpe':
                r = float(np.dot(w, expected_returns))
                v = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))
                return -(r - risk_free_rate) / v if v > 0 else 1e9
            elif objective == 'target_return':
                return float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))
            return 0.0

        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if objective == 'target_return' and target_return is not None:
            cons.append({'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return})
        bnds = tuple([(0, 1) for _ in range(n_assets)])

        res = minimize(obj, x0=np.ones(n_assets)/n_assets, method='SLSQP', bounds=bnds, constraints=cons)
        return res.x if res.success else np.ones(n_assets)/n_assets

    portfolios = {}

    # 1) BL-only (min vol)
    bl_weights = optimize_portfolio(bl_opt.bl_returns, bl_opt.bl_cov, 'min_vol')
    portfolios['BL_Only'] = {'weights': bl_weights, 'returns': bl_opt.bl_returns, 'cov': bl_opt.bl_cov}

    # 2) MBTE-only: minimize average TE to all benchmarks
    def mbte_objective(w):
        return tracking_error(w, returns_data, benchmark_data)

    res_mbte = minimize(mbte_objective, x0=np.ones(n_assets)/n_assets, method='SLSQP',
                        bounds=tuple([(0,1)]*n_assets),
                        constraints=[{'type':'eq','fun':lambda w: np.sum(w)-1}])
    portfolios['MBTE_Only'] = {'weights': res_mbte.x if res_mbte.success else np.ones(n_assets)/n_assets,
                               'returns': base_opt.mean_returns, 'cov': base_opt.cov_matrix}

    # 3) BL + MBTE combo: utility minus multi-benchmark TE penalty
    def combined_objective(w, alpha=0.6):
        bl_util = np.dot(w, bl_opt.bl_returns) - 0.5 * 3.0 * np.dot(w.T, np.dot(bl_opt.bl_cov, w))
        te_pen  = tracking_error(w, returns_data, benchmark_data)  # annualized
        return -(alpha * bl_util - (1 - alpha) * te_pen**2)

    res_comb = minimize(lambda w: combined_objective(w, 0.6), x0=np.ones(n_assets)/n_assets, method='SLSQP',
                        bounds=tuple([(0,1)]*n_assets),
                        constraints=[{'type':'eq','fun':lambda w: np.sum(w)-1}])
    portfolios['BL_MBTE'] = {'weights': res_comb.x if res_comb.success else np.ones(n_assets)/n_assets,
                             'returns': bl_opt.bl_returns, 'cov': bl_opt.bl_cov}

    # 4) BL + Single Benchmark (for comparison, same as BL min-vol)
    portfolios['BL_Single_Bench'] = {'weights': bl_weights, 'returns': bl_opt.bl_returns, 'cov': bl_opt.bl_cov}

    # Metrics
    results_df = pd.DataFrame(index=portfolios.keys())
    for name, p in portfolios.items():
        w = p['weights']
        ret, vol, shrp = portfolio_performance(w, p['returns'], p['cov'], risk_free_rate)
        te = tracking_error(w, returns_data, benchmark_data)
        results_df.loc[name, 'Expected_Return'] = ret * 100
        results_df.loc[name, 'Volatility'] = vol * 100
        results_df.loc[name, 'Sharpe_Ratio'] = shrp
        results_df.loc[name, 'Tracking_Error'] = te * 100

    print("\nMEAN-VARIANCE PORTFOLIO METRICS:")
    print("-"*50)
    print(results_df.round(3))

    allocations_df = pd.DataFrame({k: v['weights'] for k,v in portfolios.items()}, index=asset_names)
    print("\nPORTFOLIO ALLOCATIONS (%):")
    print("-"*40)
    print((allocations_df * 100).round(1))

    # Simple plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0,0].scatter(results_df['Volatility'], results_df['Expected_Return'], s=100)
    for i, name in enumerate(results_df.index):
        axes[0,0].annotate(name, (results_df.iloc[i]['Volatility'], results_df.iloc[i]['Expected_Return']))
    axes[0,0].set_xlabel('Volatility (%)'); axes[0,0].set_ylabel('Expected Return (%)'); axes[0,0].set_title('Risk-Return'); axes[0,0].grid(True)

    (allocations_df.T*100).plot(kind='bar', ax=axes[0,1], width=0.8)
    axes[0,1].set_title('Portfolio Allocations (%)'); axes[0,1].legend(bbox_to_anchor=(1.05,1), loc='upper left')

    results_df['Sharpe_Ratio'].plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Sharpe Ratios')

    results_df['Tracking_Error'].plot(kind='bar', ax=axes[1,1], color='orange')
    axes[1,1].set_title('Tracking Error vs Multi-Benchmark'); axes[1,1].set_ylabel('TE (%)')

    plt.tight_layout(); plt.show()

    return portfolios, results_df, allocations_df

def max_sharpe_analysis(returns_data, benchmark_data, risk_free_rate):
    print("\n" + "="*60)
    print("PORTFOLIO B: MAXIMUM SHARPE RATIO ANALYSIS (Real yfinance data)")
    print("="*60)

    base_opt = PortfolioOptimizer(returns_data, risk_free_rate)
    bl_opt   = BlackLittermanOptimizer(returns_data, risk_free_rate=risk_free_rate)
    mbte_opt = MBTEOptimizer(returns_data, benchmark_data, risk_free_rate=risk_free_rate)

    # BL views (same as before)
    n_assets = bl_opt.n_assets
    P = np.zeros((2, n_assets)); P[0,0] = 1; P[1,1] = 1
    Q = np.array([0.02, -0.01]); omega = np.diag([0.001, 0.001])
    bl_opt.update_views(P, Q, omega)

    def maximize_sharpe(exp_ret, cov, rfr):
        def neg_sr(w):
            r = float(np.dot(w, exp_ret)); v = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
            return -(r - rfr)/v if v > 0 else 1e9
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w)-1}]
        bnds = tuple([(0,1)]*n_assets)
        res = minimize(neg_sr, x0=np.ones(n_assets)/n_assets, method='SLSQP', bounds=bnds, constraints=cons)
        return res.x if res.success else np.ones(n_assets)/n_assets

    max_sharpe_portfolios = {}
    max_sharpe_portfolios['BL_Only'] = maximize_sharpe(bl_opt.bl_returns, bl_opt.bl_cov, risk_free_rate)

    def constrained_sharpe_mbte(w):
        r = float(np.dot(w, base_opt.mean_returns)); v = float(np.sqrt(np.dot(w.T, np.dot(base_opt.cov_matrix, w))))
        return -(r - risk_free_rate)/v if v > 0 else 1e9

    def te_constraint(w, max_te=0.08):
        te = tracking_error(w, returns_data, benchmark_data)
        return max_te - te

    res_mbte = minimize(constrained_sharpe_mbte, x0=np.ones(n_assets)/n_assets, method='SLSQP',
                        bounds=tuple([(0,1)]*n_assets),
                        constraints=[{'type':'eq','fun':lambda w: np.sum(w)-1},
                                     {'type':'ineq','fun':lambda w: te_constraint(w, 0.08)}])
    max_sharpe_portfolios['MBTE_Only'] = res_mbte.x if res_mbte.success else np.ones(n_assets)/n_assets

    def combined_obj(w, alpha=0.6):
        bl_r = float(np.dot(w, bl_opt.bl_returns))
        bl_v = float(np.sqrt(np.dot(w.T, np.dot(bl_opt.bl_cov, w))))
        bl_sr = (bl_r - risk_free_rate)/bl_v if bl_v > 0 else 0.0
        te_pen = tracking_error(w, returns_data, benchmark_data)
        return -(alpha*bl_sr - (1-alpha)*te_pen)

    res_comb = minimize(lambda w: combined_obj(w, 0.6), x0=np.ones(n_assets)/n_assets, method='SLSQP',
                        bounds=tuple([(0,1)]*n_assets), constraints=[{'type':'eq','fun':lambda w: np.sum(w)-1}])
    max_sharpe_portfolios['BL_MBTE'] = res_comb.x if res_comb.success else np.ones(n_assets)/n_assets
    max_sharpe_portfolios['BL_Single_Bench'] = maximize_sharpe(bl_opt.bl_returns, bl_opt.bl_cov, risk_free_rate)

    sharpe_results = pd.DataFrame(index=max_sharpe_portfolios.keys())
    for name, w in max_sharpe_portfolios.items():
        if 'BL' in name:
            exp_ret, cov = bl_opt.bl_returns, bl_opt.bl_cov
        else:
            exp_ret, cov = base_opt.mean_returns, base_opt.cov_matrix
        ret, vol, shrp = portfolio_performance(w, exp_ret, cov, risk_free_rate)
        te = tracking_error(w, returns_data, benchmark_data)
        sharpe_results.loc[name, ['Expected_Return','Volatility','Sharpe_Ratio','Tracking_Error']] = [ret*100, vol*100, shrp, te*100]

    print("\nMAXIMUM SHARPE RATIO PORTFOLIOS:")
    print("-"*45); print(sharpe_results.round(3))

    baseline = sharpe_results.loc['BL_Only','Sharpe_Ratio']
    print("\nSHARPE RATIO IMPROVEMENTS:")
    print("-"*35)
    for name in sharpe_results.index:
        imp = sharpe_results.loc[name,'Sharpe_Ratio'] - baseline
        pct = (imp/baseline*100) if pd.notna(baseline) and baseline != 0 else np.nan
        print(f"{name}: {imp:+.3f} ({pct:+.1f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sharpe_results['Sharpe_Ratio'].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Maximum Sharpe Ratios')

    axes[1].scatter(sharpe_results['Volatility'], sharpe_results['Expected_Return'], s=150)
    for i, name in enumerate(sharpe_results.index):
        axes[1].annotate(name, (sharpe_results.iloc[i]['Volatility'], sharpe_results.iloc[i]['Expected_Return']))
    axes[1].set_xlabel('Volatility (%)'); axes[1].set_ylabel('Expected Return (%)'); axes[1].set_title('Risk-Return'); axes[1].grid(True)

    te_vs = sharpe_results[['Tracking_Error','Sharpe_Ratio']]
    axes[2].scatter(te_vs['Tracking_Error'], te_vs['Sharpe_Ratio'], s=150)
    for i, name in enumerate(te_vs.index):
        axes[2].annotate(name, (te_vs.iloc[i]['Tracking_Error'], te_vs.iloc[i]['Sharpe_Ratio']))
    axes[2].set_xlabel('Tracking Error (%)'); axes[2].set_ylabel('Sharpe Ratio'); axes[2].set_title('TE vs Sharpe'); axes[2].grid(True)

    plt.tight_layout(); plt.show()
    return max_sharpe_portfolios, sharpe_results

def higher_moments_analysis(returns_data, benchmark_data, risk_free_rate):
    print("\n" + "="*60)
    print("PORTFOLIO C: HIGHER MOMENTS ANALYSIS (Real yfinance data)")
    print("="*60)

    asset_names = list(returns_data.columns)
    base_opt = PortfolioOptimizer(returns_data, risk_free_rate)
    bl_opt   = BlackLittermanOptimizer(returns_data, risk_free_rate=risk_free_rate)
    mbte_opt = MBTEOptimizer(returns_data, benchmark_data, risk_free_rate=risk_free_rate)

    # Views (as before)
    n_assets = bl_opt.n_assets
    P = np.zeros((2, n_assets)); P[0,0]=1; P[1,1]=1
    Q = np.array([0.02, -0.01]); omega = np.diag([0.001, 0.001])
    bl_opt.update_views(P, Q, omega)

    def calculate_portfolio_moments(w, ret_df):
        port = ret_df.dot(w)
        return float(skew(port)), float(kurtosis(port, fisher=True))

    def higher_moment_objective(w, exp_ret, cov, ret_df, skew_penalty=0.1, kurt_penalty=0.05):
        r = float(np.dot(w, exp_ret))
        var = float(np.dot(w.T, np.dot(cov, w)))
        mv_util = r - 0.5 * 3.0 * var
        port = ret_df.dot(w)
        ps = float(skew(port)); pk = float(kurtosis(port, fisher=True))
        skew_term = skew_penalty * min(0.0, ps)**2
        kurt_term = kurt_penalty * max(0.0, pk)**2
        return -(mv_util - skew_term - kurt_term)

    n_assets = len(asset_names)

    hm_ports = {}

    # 1) BL with higher-moment penalties
    res_bl_hm = minimize(lambda w: higher_moment_objective(w, bl_opt.bl_returns, bl_opt.bl_cov, returns_data.values),
                         x0=np.ones(n_assets)/n_assets, method='SLSQP',
                         bounds=tuple([(0,1)]*n_assets),
                         constraints=[{'type':'eq','fun':lambda w: np.sum(w)-1}])
    hm_ports['BL_Higher_Moments'] = res_bl_hm.x if res_bl_hm.success else np.ones(n_assets)/n_assets

    # 2) MBTE + higher-moments
    def mbte_hm_obj(w):
        base_u = -higher_moment_objective(w, base_opt.mean_returns, base_opt.cov_matrix, returns_data.values)
        te_pen = tracking_error(w, returns_data, benchmark_data)
        return -(0.7*base_u - 0.3*te_pen**2)
    res_mbte_hm = minimize(mbte_hm_obj, x0=np.ones(n_assets)/n_assets, method='SLSQP',
                           bounds=tuple([(0,1)]*n_assets),
                           constraints=[{'type':'eq','fun':lambda w: np.sum(w)-1}])
    hm_ports['MBTE_Higher_Moments'] = res_mbte_hm.x if res_mbte_hm.success else np.ones(n_assets)/n_assets

    # 3) Combined BL+MBTE + higher-moments
    def comb_hm_obj(w, alpha=0.65):
        bl_u = -higher_moment_objective(w, bl_opt.bl_returns, bl_opt.bl_cov, returns_data.values, 0.15, 0.08)
        te_pen = tracking_error(w, returns_data, benchmark_data)
        return -(alpha*bl_u - (1-alpha)*te_pen**2)
    res_comb_hm = minimize(lambda w: comb_hm_obj(w, 0.65), x0=np.ones(n_assets)/n_assets, method='SLSQP',
                           bounds=tuple([(0,1)]*n_assets),
                           constraints=[{'type':'eq','fun':lambda w: np.sum(w)-1}])
    hm_ports['BL_MBTE_Higher_Moments'] = res_comb_hm.x if res_comb_hm.success else np.ones(n_assets)/n_assets

    # 4) Traditional MV baseline
    def mv_obj(w):
        r = float(np.dot(w, bl_opt.bl_returns)); v = float(np.dot(w.T, np.dot(bl_opt.bl_cov, w)))
        return -(r - 0.5 * 3.0 * v)
    res_mv = minimize(mv_obj, x0=np.ones(n_assets)/n_assets, method='SLSQP',
                      bounds=tuple([(0,1)]*n_assets),
                      constraints=[{'type':'eq','fun':lambda w: np.sum(w)-1}])
    hm_ports['Traditional_MV'] = res_mv.x if res_mv.success else np.ones(n_assets)/n_assets

    # Metrics
    hm_results = pd.DataFrame(index=hm_ports.keys())
    for name, w in hm_ports.items():
        exp_ret, cov = (bl_opt.bl_returns, bl_opt.bl_cov) if ('BL' in name and name!='Traditional_MV') else (base_opt.mean_returns, base_opt.cov_matrix)
        ret, vol, shrp = portfolio_performance(w, exp_ret, cov, risk_free_rate)
        te = tracking_error(w, returns_data, benchmark_data)
        ps, pk = calculate_portfolio_moments(w, returns_data)
        hm_results.loc[name, ['Expected_Return','Volatility','Sharpe_Ratio','Tracking_Error','Skewness','Excess_Kurtosis']] = \
            [ret*100, vol*100, shrp, te*100, ps, pk]

    print("\nHIGHER MOMENTS PORTFOLIO ANALYSIS:")
    print("-"*50); print(hm_results.round(3))

    hm_alloc = pd.DataFrame({k:v for k,v in hm_ports.items()}, index=asset_names)
    print("\nHIGHER MOMENTS PORTFOLIO ALLOCATIONS (%):")
    print("-"*50); print((hm_alloc*100).round(1))

    # Quick visuals
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    bubble = np.abs(hm_results['Skewness'])*300 + 50
    sc = axes[0,0].scatter(hm_results['Volatility'], hm_results['Expected_Return'], s=bubble, alpha=0.7, c=hm_results['Excess_Kurtosis'], cmap='RdYlBu')
    for i, name in enumerate(hm_results.index):
        axes[0,0].annotate(name, (hm_results.iloc[i]['Volatility'], hm_results.iloc[i]['Expected_Return']), xytext=(5,5), textcoords='offset points', fontsize=8)
    axes[0,0].set_xlabel('Volatility (%)'); axes[0,0].set_ylabel('Expected Return (%)'); axes[0,0].set_title('Risk-Return (size=|skew|, color=kurt)'); plt.colorbar(sc, ax=axes[0,0])

    hm_results['Skewness'].plot(kind='bar', ax=axes[0,1]); axes[0,1].set_title('Portfolio Skewness'); axes[0,1].axhline(0, color='k', ls='--', alpha=0.4)
    hm_results['Excess_Kurtosis'].plot(kind='bar', ax=axes[0,2]); axes[0,2].set_title('Portfolio Excess Kurtosis'); axes[0,2].axhline(0, color='k', ls='--', alpha=0.4)

    (hm_alloc.T*100).plot(kind='bar', ax=axes[1,0], width=0.8); axes[1,0].set_title('Allocations (%)'); axes[1,0].legend(bbox_to_anchor=(1.05,1), loc='upper left')

    axes[1,1].scatter(hm_results['Skewness'], hm_results['Sharpe_Ratio'], s=100, alpha=0.7)
    for i, name in enumerate(hm_results.index):
        axes[1,1].annotate(name, (hm_results.iloc[i]['Skewness'], hm_results.iloc[i]['Sharpe_Ratio']), xytext=(5,5), textcoords='offset points', fontsize=8)
    axes[1,1].set_xlabel('Skewness'); axes[1,1].set_ylabel('Sharpe'); axes[1,1].set_title('Sharpe vs Skew'); axes[1,1].grid(True, alpha=0.3)

    # (Optional) omit higher-moment frontier for runtime brevity

    plt.tight_layout(); plt.show()

    return hm_ports, hm_results, hm_alloc

# -----------------------------
# Master runner
# -----------------------------
def run_complete_analysis(start='2015-01-01', end=None):
    print("COMPREHENSIVE PORTFOLIO OPTIMIZATION ANALYSIS (yfinance)")
    print("="*80)

    assets_ret, sector_ret, style_ret, bench_ret, rfr_annual = load_market_data(start, end)

    print(f"\nRisk-free (13w T-Bill, annualized): {rfr_annual:.2%}")
    print(f"Benchmarks loaded: {', '.join(bench_ret.columns)}")
    print(f"Assets loaded: {len(assets_ret.columns)} tickers")

    mv_portfolios, mv_results, mv_alloc = mean_variance_analysis(assets_ret, bench_ret, rfr_annual)
    sharpe_ports, sharpe_results = max_sharpe_analysis(assets_ret, bench_ret, rfr_annual)
    hm_ports, hm_results, hm_alloc = higher_moments_analysis(assets_ret, bench_ret, rfr_annual)

    print("\n" + "="*80)
    print("SUMMARY COMPARISON ACROSS ALL METHODS")
    print("="*80)

    best_mv = mv_results.loc[mv_results['Sharpe_Ratio'].idxmax()]
    best_sh = sharpe_results.loc[sharpe_results['Sharpe_Ratio'].idxmax()]
    best_hm = hm_results.loc[hm_results['Sharpe_Ratio'].idxmax()]

    summary = pd.DataFrame({
        'Best_MeanVariance': best_mv,
        'Best_MaxSharpe': best_sh,
        'Best_HigherMoments': best_hm
    }).T

    cols = ['Expected_Return','Volatility','Sharpe_Ratio','Tracking_Error']
    print("\nBEST PORTFOLIO FROM EACH METHOD:")
    print("-"*45); print(summary[cols].round(3))

    return {
        'mean_variance': (mv_portfolios, mv_results, mv_alloc),
        'max_sharpe': (sharpe_ports, sharpe_results),
        'higher_moments': (hm_ports, hm_results, hm_alloc),
        'risk_free_annual': rfr_annual
    }

if __name__ == "__main__":
    results = run_complete_analysis(start='2015-01-01', end=None)
