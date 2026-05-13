# Monte Carlo Option Pricing and Stochastic Volatility  
*A quantitative framework for option pricing, stochastic volatility modeling, and variance reduction*

---

## Overview

Monte Carlo methods are widely used for pricing derivatives when closed-form solutions are unavailable, but they can suffer from high variance and computational inefficiency.

In this project, I build a Monte Carlo pricing framework for European options and extend it in two key directions:

- Introduce **stochastic volatility** using the Heston model  
- Improve efficiency using **variance reduction via control variates**

This combines realistic modeling of market behavior with improved numerical performance.

---

## Model

### Black–Scholes Baseline

\[
dS_t = r S_t dt + \sigma S_t dW_t
\]

---

### Heston Stochastic Volatility Model

\[
dS_t = r S_t dt + \sqrt{v_t} S_t dW_t^{(1)}
\]

\[
dv_t = \kappa(\theta - v_t) dt + \xi \sqrt{v_t} dW_t^{(2)}
\]

\[
\mathbb{E}[dW_t^{(1)} dW_t^{(2)}] = \rho\, dt
\]

---

## Methodology

### Monte Carlo Pricing
- Simulated asset paths under risk-neutral dynamics  
- Computed discounted payoff:
  \[
  C = e^{-rT} \mathbb{E}[(S_T - K)^+]
  \]
- Estimated prices using sample averages  

---

### Stochastic Volatility (Heston)
- Simulated variance process using Euler discretization  
- Used exponential scheme for price evolution  
- Incorporated correlation between price and volatility  

---

### Variance Reduction (Control Variate)

Used the control variate:

\[
Y = e^{-rT} S_T
\]

with known expectation:

\[
\mathbb{E}[Y] = S_0
\]

Constructed estimator:

\[
Z = X + c(Y - S_0)
\]

which reduces variance without biasing the estimate.

---

## Results

### Monte Carlo + Control Variate

- Variance reduction factor: **~2.9x**  
- Confidence intervals significantly tightened  
- Same accuracy achieved with fewer simulation paths  

---

### Heston Calibration

- Initial RMSE: **0.0762**  
- Final RMSE: **0.0288**  

#### Calibrated Parameters

| Parameter | Value |
|----------|------|
| \( \kappa \) | 20.0000 |
| \( \theta \) | 0.0550 |
| \( v_0 \) | 0.0709 |
| \( \xi \) | 0.7799 |
| \( \rho \) | -0.7061 |

---

## Key Insights

- Stochastic volatility models capture market features such as skew and smile  
- Monte Carlo methods are flexible but require variance reduction for efficiency  
- Control variates significantly improve estimator precision  
- Calibration is an inverse problem and may be ill-conditioned  

---

## Limitations

- Euler discretization introduces bias  
- Calibration may not uniquely identify parameters  
- No jump processes or stochastic interest rates  
- Assumes continuous price evolution  

---

## Repository Structure

- `montecarlo.py` — Monte Carlo pricing with control variate  
- `hestonproject.py` — stochastic volatility simulation and calibration  
- `Volatility Notes.pdf` — stochastic volatility and Monte Carlo methodology  
- `Calibration.pdf` — calibration algorithm and numerical results  

---

## Additional Resources

- Detailed derivations and results are provided in the included PDFs
