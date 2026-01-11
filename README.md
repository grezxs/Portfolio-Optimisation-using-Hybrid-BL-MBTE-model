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

Repository Structure
├── data/               # Market and benchmark data
├── notebooks/          # Research and analysis notebooks
├── src/                # Model implementation and utilities
├── results/            # Performance outputs and diagnostics
├── requirements.txt    # Dependencies
└── README.md

Disclaimer

This project is for educational and research purposes only. It does not constitute investment advice or a production-ready trading system.

If you want, I can:

Tighten this for ATS-style recruiter scanning

Rewrite it for front-office quant / asset management roles

Add equations, references, or diagrams

Convert this into an academic-paper-aligned README

Just tell me the target role.

Rewrite it for front-office quant / asset management roles

Below is a front-office quant / asset-management–oriented rewrite of the README. The tone emphasizes portfolio construction relevance, risk control, benchmark awareness, and decision-making under institutional constraints, while remaining technically credible and restrained.

Multi-Benchmark Tracking Error–Black-Litterman (MBTE-BL) Portfolio Optimization
Executive Overview

This repository implements a Multi-Benchmark Tracking Error–Black-Litterman (MBTE-BL) portfolio construction framework designed for institutional asset management and front-office quantitative research. The model addresses benchmark dependency and instability arising from single-benchmark optimization by incorporating multiple reference benchmarks into the portfolio optimization process.

The framework integrates Black-Litterman Bayesian return estimation with multi-benchmark tracking error control, aligning portfolio construction with how professional mandates are evaluated in practice.

Investment Rationale

Institutional portfolios are rarely assessed against a single benchmark. Performance is typically evaluated across multiple indices, styles, or mandates, exposing single-benchmark optimization to:

Benchmark overfitting

Unstable allocations

Poor out-of-sample behavior

The MBTE-BL framework reflects real-world portfolio construction constraints by:

Anchoring expected returns via Black-Litterman

Controlling relative risk across several benchmarks simultaneously

Improving robustness in dynamic market environments

Framework Overview
Black-Litterman (BL)

Bayesian combination of equilibrium returns and investor views

Reduces estimation error in expected returns

Provides structured, interpretable subjectivity

Multi-Benchmark Tracking Error (MBTE)

Extends traditional tracking error minimization

Manages active risk relative to multiple benchmarks

Enhances diversification across evaluation criteria

Integrated MBTE-BL Model

BL generates stabilized expected returns

MBTE governs active risk across benchmark set

Optimization balances alpha expression with benchmark discipline

Data & Universe

Assets: 50 S&P 500 constituent equities

Factors: 5 sector-level factors

Benchmarks: 9 ETFs representing multiple reference indices

Data Source: yfinance

Empirical Findings

MBTE-BL outperforms the standard Black-Litterman model

Performance remains below the standalone MBTE specification

Results highlight the trade-off between Bayesian structure and empirical flexibility

The findings suggest that while BL stabilizes return estimation, benchmark weighting and allocation sensitivity remain key drivers of performance.

Risk Considerations & Limitations

Sensitivity to benchmark weighting schemes

Equal-weight treatment of benchmarks

Dependence on prior specification and view confidence

Constraint-driven allocation concentration

These limitations are consistent with real-world portfolio construction challenges and inform future refinements.
