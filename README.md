# Portfolio Optimization Using a Hybrid BL–MBTE Model

Quantitative research implementation of a **Multi-Benchmark Tracking Error–Black-Litterman (MBTE-BL)** portfolio optimization framework.  
This project combines Bayesian return estimation with adaptive multi-benchmark tracking to address benchmark dependency, overfitting, and diversification limits in traditional portfolio construction.

---

## Overview

This repository presents a quantitative research implementation of a **Multi-Benchmark Tracking Error–Black-Litterman (MBTE-BL)** portfolio optimization framework.

Traditional portfolio construction methods often rely on a **single benchmark**, which can lead to:
- Excessive benchmark dependency  
- Overfitting to a reference index  
- Artificial constraints on diversification  

By integrating the **Black-Litterman (BL)** model with a **Multi-Benchmark Tracking Error (MBTE)** approach, this framework aims to balance **structured Bayesian priors** with **empirical robustness** in environments where portfolios are evaluated against multiple benchmarks.

---

## Motivation

Classical mean–variance and benchmark-relative optimization frameworks assume a single reference benchmark. In practice, institutional portfolios are frequently evaluated against **multiple indices, styles, or mandates**, making single-benchmark assumptions restrictive and unstable.

This project is motivated by the need to address:

- Benchmark dependency in traditional optimization  
- Overfitting to a single reference index  
- Limited robustness in multi-evaluation environments  

The MBTE-BL framework is designed to mitigate these issues while preserving interpretability and disciplined portfolio construction.

---

## Methodology

### Core Components

#### Black-Litterman (BL) Model
- Bayesian framework combining equilibrium returns with investor views  
- Reduces estimation error in expected returns  
- Produces more stable and interpretable return estimates  

#### Multi-Benchmark Tracking Error (MBTE)
- Extends traditional tracking error minimization to **multiple benchmarks**  
- Enhances diversification  
- Improves robustness to benchmark selection  

### Integrated Framework: MBTE-BL

The combined framework:
- Uses BL to generate stabilized expected returns  
- Applies MBTE to control deviations across multiple benchmarks  
- Seeks improved portfolio stability and risk-adjusted performance  

---

## Data

- **Assets:** 50 stocks selected from the S&P 500 Index  
- **Factors:** 5 sector-based factors  
- **Benchmarks:** 9 ETFs representing multiple reference indices  
- **Data Source:** All market data retrieved using `yfinance`  

---

## Results Summary

Key empirical findings:
- The **MBTE-BL framework outperformed the standard Black-Litterman model**
- Performance remained **inferior to the standalone MBTE model**
- Results highlight trade-offs between:
  - Bayesian structure (BL)
  - Empirical flexibility (MBTE)

These findings suggest that while integration improves robustness relative to BL, further refinement is required to fully capture the advantages of MBTE.

---

## Limitations

- Sensitivity to asset allocation constraints  
- Equal-weight treatment of benchmarks  
- Dependence on prior specification within the BL framework  

---

## Future Work

Potential extensions include:
- Replacing equilibrium priors with free-float market capitalization weights  
- Introducing non-uniform benchmark weights  
- Applying adaptive benchmark weighting methods (e.g., Thompson Sampling)  
- Robustness testing across different market regimes  
- Expanding factor and asset universes  

---

## Live Application

An interactive dashboard demonstrating the framework is available here:

👉 **Live App:** https://huggingface.co/spaces/gchandra19/BL-MBTE-Dashboard  

---

## Disclaimer

This project is for **educational and research purposes only**.  
It does **not** constitute investment advice or a production-ready trading system.
