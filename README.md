# Portfolio-Optimisation-using-Hybrid-BL-MBTE-model
Quantitative research implementation of a Multi-Benchmark Tracking Error–Black-Litterman (MBTE-BL) portfolio optimization framework. Combines Bayesian return estimation with adaptive multi-benchmark tracking to address benchmark dependency, overfitting, and diversification limits in traditional portfolio construction.


Overview

This repository presents a quantitative research implementation of a Multi-Benchmark Tracking Error–Black-Litterman (MBTE-BL) portfolio optimization framework. The project addresses key limitations of traditional portfolio construction methods that rely on a single benchmark, which often results in benchmark dependency, overfitting, and constrained diversification.

By integrating Black-Litterman (BL) Bayesian return estimation with a Multi-Benchmark Tracking Error (MBTE) approach, this framework seeks to balance structured subjective views with empirical adaptiveness in environments where portfolios are evaluated against multiple reference indices.



Motivation

Classical mean–variance and benchmark-relative optimization frameworks assume a single reference benchmark. In practice, institutional portfolios are frequently assessed against multiple indices, styles, or mandates, making single-benchmark assumptions restrictive and unstable.

This project is motivated by:

Benchmark dependency in traditional optimization

Overfitting to a single reference index

Limited robustness in dynamic, multi-evaluation environments

The MBTE-BL framework is designed to mitigate these issues while maintaining interpretability and disciplined portfolio construction.



Methodology
Core Components

Black-Litterman Model

Bayesian framework combining equilibrium returns with investor views

Reduces estimation error in expected returns

Multi-Benchmark Tracking Error (MBTE)

Extends tracking error minimization across multiple benchmarks

Enhances diversification and benchmark robustness

Integrated Framework (MBTE-BL)

The combined model:

Uses BL to generate stabilized return expectations

Applies MBTE to control deviations across multiple benchmarks

Seeks improved portfolio stability and risk-adjusted performance



Data

Assets: 50 stocks selected from the S&P 500 Index

Factors: 5 sector-based factors

Benchmarks: 9 ETFs representing multiple reference indices

Source: All data retrieved using yfinance



Results Summary

The MBTE-BL framework outperformed the standard Black-Litterman model

Performance remained inferior to the standalone MBTE model

Results highlight trade-offs between Bayesian structure and empirical flexibility

These findings suggest that while integration improves robustness relative to BL, further refinement is required to fully capture the advantages of MBTE.



Limitations

Sensitivity to asset allocation constraints

Equal-weight treatment of benchmarks

Dependence on prior specification within the BL framework



Future Work

Potential extensions include:

Replacing equilibrium priors with free-float market capitalization

Introducing non-uniform benchmark weights

Applying adaptive benchmark weighting methods (e.g., Thompson Sampling)

Robustness testing across market regimes

Expanded factor and asset universes



Intended Audience

This repository is intended for:

Recruiters and hiring managers evaluating quantitative finance capability

Asset management and portfolio construction professionals

Students and researchers interested in applied portfolio optimization

The project emphasizes research discipline, transparency, and reproducibility over production optimization.



Disclaimer

This project is for educational and research purposes only. It does not constitute investment advice or a production-ready trading system.


