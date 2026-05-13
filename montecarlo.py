import matplotlib.pyplot as plt
import numpy as np


np.random.seed(1)


def hestonpath(S0, T, r, steps, kappa, theta, v0, rho, xi, Npaths, return_vol=False):
    dt = T / steps
    prices = np.zeros((Npaths, steps + 1))
    vars_ = np.zeros((Npaths, steps + 1))

    St = np.full(Npaths, S0, dtype=float)
    vt = np.full(Npaths, v0, dtype=float)

    prices[:, 0] = St
    vars_[:, 0] = vt

    cov = np.array([[1.0, rho], [rho, 1.0]])

    for i in range(steps):
        Z = np.random.multivariate_normal([0.0, 0.0], cov, size=Npaths)
        Z1 = Z[:, 0]
        Z2 = Z[:, 1]

        vt_pos = np.maximum(vt, 0.0)
        vt = (
            vt_pos
            + kappa * (theta - vt_pos) * dt
            + xi * np.sqrt(vt_pos) * np.sqrt(dt) * Z2
        )
        vt = np.maximum(vt, 0.0)
        St = St * np.exp((r - 0.5 * vt_pos) * dt + np.sqrt(vt_pos) * np.sqrt(dt) * Z1)
        prices[:, i + 1] = St
        vars_[:, i + 1] = vt
    if return_vol:
        return prices, vars_
    return prices


S0 = 200
T = 1.0
r = 0.04
steps = 252
kappa = 2.0
theta = 0.04
v0 = 0.04
rho = -0.473
xi = 0.6
Npaths = 1000000
K = 210
S = hestonpath(S0, T, r, steps, kappa, theta, v0, rho, xi, Npaths)

# plt.figure(figsize=(10, 5))
# plt.plot(S.T)
# plt.title("Heston Model Price Paths")
# plt.xlabel("Step")
# plt.ylabel("Price")
# plt.show()

ST = S[:, -1]
disc_payoffs = np.exp(-r * T) * np.maximum(ST - K, 0.0)
price_est = np.mean(disc_payoffs)
var_disc_payoff = np.var(disc_payoffs, ddof=1)
var_mc_estimator = var_disc_payoff / Npaths
se_mc = np.sqrt(var_mc_estimator)
print("Call Premium price estimate:", price_est)
print("Var(discounted payoff):", var_disc_payoff)
print("Var(MC estimator):", var_mc_estimator)
print("Std error:", se_mc)


Y = np.zeros(Npaths)
Y = np.exp(-r * T) * ST
X = disc_payoffs
cov_XY = np.cov(X, Y, ddof=1)[0, 1]
var_Y = np.var(Y, ddof=1)
c = -cov_XY / var_Y
control = X + c * (Y - S0)
print("Control variate price:", np.mean(control))
print("Var(control variate):", np.var(control, ddof=1))
print("Variance reduction factor:", np.var(X, ddof=1) / np.var(control, ddof=1))


z = 1.96

# --- Plain MC CI ---
mean_X = np.mean(disc_payoffs)
var_X = np.var(disc_payoffs, ddof=1)
se_X = np.sqrt(var_X / Npaths)
ci_X = (mean_X - z * se_X, mean_X + z * se_X)

# --- Control variate CI ---
mean_Z = np.mean(control)
var_Z = np.var(control, ddof=1)
se_Z = np.sqrt(var_Z / Npaths)
ci_Z = (mean_Z - z * se_Z, mean_Z + z * se_Z)

print("Without Control 95% CI:", ci_X)
print("Control Variate 95% CI:", ci_Z)
print("CI width (no control):", ci_X[1] - ci_X[0])
print("CI width (control):", ci_Z[1] - ci_Z[0])
print("CI shrink factor:", (ci_X[1] - ci_X[0]) / (ci_Z[1] - ci_Z[0]))
