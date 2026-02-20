# Statistics Examples

This directory contains 10 standalone Python scripts demonstrating key statistical concepts and methods.

## Files Overview

### 01_probability_review.py (~250 lines)
- Normal, Binomial, Poisson, and Exponential distributions
- PDF/CDF plotting and calculations
- Central Limit Theorem demonstration
- Law of Large Numbers
- Random sampling with numpy/scipy.stats

### 02_sampling_estimation.py (~260 lines)
- Simple random sampling
- Stratified sampling (with comparison to simple random)
- Bootstrap estimation and resampling
- Confidence intervals for mean
- Bias and variance of estimators
- Maximum Likelihood Estimation (MLE) for normal distribution

### 03_hypothesis_testing.py (~280 lines)
- One-sample, two-sample, and paired t-tests
- Chi-square tests (goodness of fit and independence)
- One-way ANOVA
- p-value computation
- Statistical power analysis
- Multiple testing correction (Bonferroni and FDR)

### 04_regression_analysis.py (~260 lines)
- OLS regression from scratch (simple and multiple)
- Polynomial regression with model selection
- Residual analysis (normality, heteroscedasticity, autocorrelation)
- R-squared and adjusted R-squared
- Confidence and prediction intervals
- AIC/BIC model comparison

### 05_bayesian_basics.py (~240 lines)
- Bayes' theorem with discrete examples
- Beta-Binomial conjugate prior
- Normal-Normal conjugate prior
- Prior influence on posterior
- Credible vs confidence intervals
- Sequential Bayesian updating

### 06_bayesian_inference.py (~280 lines)
- Metropolis-Hastings MCMC algorithm
- Gibbs sampling for bivariate normal
- Bayesian linear regression (closed-form posterior)
- Convergence diagnostics (trace plots, Gelman-Rubin R̂)
- Posterior sampling and credible intervals

### 07_glm.py (~300 lines)
- Logistic regression from scratch (MLE optimization)
- Poisson regression
- Link functions (logit, probit, log, identity)
- Deviance analysis
- AIC/BIC model comparison for GLMs
- Likelihood ratio tests

### 08_time_series.py (~270 lines)
- Moving average smoothing
- Exponential smoothing (different alpha values)
- Autocorrelation function (ACF)
- Stationarity testing concepts
- AR, MA, and ARMA models
- Simple forecasting methods (naive, mean, AR)

### 09_multivariate.py (~300 lines)
- PCA from scratch (eigendecomposition)
- PCA for dimensionality reduction
- Multivariate normal distribution
- Mahalanobis distance and outlier detection
- Canonical correlation analysis (CCA concept)

### 10_nonparametric.py (~330 lines)
- Mann-Whitney U test (non-normal two-sample)
- Wilcoxon signed-rank test (paired non-normal)
- Kruskal-Wallis test (nonparametric ANOVA)
- Kolmogorov-Smirnov test (goodness of fit, two-sample)
- Kernel density estimation (KDE with different bandwidths)
- Bootstrap confidence intervals (percentile method)

## Usage

Each script is self-contained and can be run independently:

```bash
python 01_probability_review.py
python 02_sampling_estimation.py
# ... and so on
```

All scripts:
- Print detailed results to stdout
- Generate synthetic data (no external data files needed)
- Include optional matplotlib visualizations (saved to /tmp/)
- Use numpy and scipy.stats for computations
- Are ~150-330 lines each
- Include docstrings and section headers

## Dependencies

- Python 3.7+
- numpy
- scipy
- matplotlib (optional, for plots)

Install dependencies:
```bash
pip install numpy scipy matplotlib
```

## Topics Covered

1. **Probability**: Distributions, CLT, LLN
2. **Sampling**: Random, stratified, bootstrap
3. **Hypothesis Testing**: t-tests, ANOVA, chi-square, power, multiple testing
4. **Regression**: OLS, polynomial, residuals, intervals
5. **Bayesian Basics**: Conjugate priors, credible intervals
6. **Bayesian Inference**: MCMC, Gibbs, Bayesian regression
7. **GLM**: Logistic, Poisson, link functions, deviance
8. **Time Series**: MA, ES, ACF, AR/MA/ARMA, forecasting
9. **Multivariate**: PCA, Mahalanobis, CCA
10. **Nonparametric**: Rank tests, KS, KDE, bootstrap

## License

MIT License (code examples)
