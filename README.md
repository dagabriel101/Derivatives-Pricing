# Monte Carlo Option Pricing and Stochastic Volatility  
*A brief work on option pricing, stochastic volatility modeling, and variance reduction*

---

## Overview

Monte Carlo methods may be used to price derivatives, but may suffer from high levels of variance and computational inefficiency

In this project, I build a Monte Carlo pricing model for European options and had two main focuses:

- Utilized the Heston framework as a way to implement stochastic volatility models
- Utilize control variates in order to reduce variance

---

## Model

### Black–Scholes Baseline

$$
dS_t = r S_t dt + \sigma S_t dW_t
$$

---

### Heston Stochastic Volatility Model

$$
dS_t = r S_t dt + \sqrt{v_t} S_t dW_t^{(1)}
$$

$$
dv_t = \kappa(\theta - v_t) dt + \xi \sqrt{v_t} dW_t^{(2)}
$$

$$
\mathbb{E}[dW_t^{(1)} dW_t^{(2)}] = \rho\, dt
$$

---

## Methodology

### Monte Carlo Pricing
- Simulated asset paths under risk-neutral dynamics  
- Estimated option prices via discounted payoffs  

$$
C = e^{-rT} \mathbb{E}[(S_T - K)^+]
$$

---

### Stochastic Volatility
- Simulated variance using Euler discretization  
- Used exponential scheme for price evolution  
- Incorporated correlation between price and volatility  

---

### Variance Reduction (Control Variate)

Used the control variate:

$$
Y = e^{-rT} S_T
$$

with known expectation:

$$
\mathbb{E}[Y] = S_0
$$

Constructed estimator:

$$
Z = X + c(Y - S_0)
$$

---

## Results

### Variance Reduction

- Variance reduction factor: **~2.9x**  
- Confidence intervals shrank  
- Same accuracy achieved with fewer simulations  

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

- Stochastic volatility captures skew and smile  
- Monte Carlo requires variance reduction for efficiency  
- Control variates significantly improve estimator precision  
- Calibration is an inverse problem and may be ill-conditioned  

---

## Limitations

- Euler discretization introduces bias  
- Parameters may not be uniquely identified  
- No jump processes or stochastic rates  
- Continuous-time assumption  

---

## Extensions

- Milstein or exact simulation  
- Jump-diffusion models  
- Implied volatility surface calibration  
- Bayesian methods  

---

## Repository Structure

- `montecarlo.py` — Monte Carlo pricing with control variate  
- `hestonproject.py` — stochastic volatility simulation and calibration  
- `Volatility Notes.pdf` — stochastic volatility theory and results  
- `Calibration.pdf` — calibration algorithm and numerical results  

---

## Additional Resources

- Full derivations and results included in PDFs
