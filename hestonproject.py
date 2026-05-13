import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

np.random.seed(134)


def hestonmodel(S, K, r, T, kappa, theta, v0, rho, steps, paths, xi, Z1, Z2):
    T = float(T)
    steps = int(steps)
    dt = T / steps
    St = np.full(paths, float(S), dtype=float)
    vt = np.full(paths, float(v0), dtype=float)
    rho = float(rho)
    Z2c = rho * Z1 + np.sqrt(1.0 - rho**2) * Z2
    for i in range(steps):
        vt_pos = np.maximum(vt, 0.0)
        vt = (
            vt_pos
            + kappa * (theta - vt_pos) * dt
            + xi * np.sqrt(vt_pos) * np.sqrt(dt) * Z2c[i, :]
        )
        vt = np.maximum(vt, 0.0)
        St = St * np.exp(
            (r - 0.5 * vt_pos) * dt + np.sqrt(vt_pos) * np.sqrt(dt) * Z1[i, :]
        )
    K = np.asarray(K, dtype=float)
    payoff = np.maximum(St[:, None] - K[None, :], 0.0)
    prices = np.exp(-r * T) * payoff.mean(axis=0)
    return prices


def BS_call(S, K, r, T, sigma):
    S = float(S)
    K = float(K)
    r = float(r)
    T = float(T)
    sigma = float(sigma)
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bisection_iv(S, K, r, T, sig_lo, sig_hi, price, tol=1e-7, maxiter=200):
    f_lo = BS_call(S, K, r, T, sig_lo) - price
    f_hi = BS_call(S, K, r, T, sig_hi) - price
    if not (f_lo < 0 and f_hi > 0):
        return np.nan
    lo, hi = float(sig_lo), float(sig_hi)
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        f_mid = BS_call(S, K, r, T, mid) - price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def steps_by_T(T):
    # simple rule: ~400 steps per year, min 60
    return int(max(60, round(400.0 * float(T))))


def make_Z_by_T(Tvec, paths, seed=134):
    rng = np.random.default_rng(seed)
    Z_by_T = {}
    for T in np.unique(Tvec):
        st = steps_by_T(T)
        Z1 = rng.standard_normal((st, paths))
        Z2 = rng.standard_normal((st, paths))
        Z_by_T[float(T)] = (Z1, Z2, st)
    return Z_by_T


def model_prices_surface(S, Tvec, Kvec, r, kappa, theta, v0, xi, rho, paths, Z_by_T):
    C_model = np.empty_like(Kvec, dtype=float)
    for T in np.unique(Tvec):
        idx = np.where(Tvec == T)[0]
        Z1, Z2, st = Z_by_T[float(T)]
        C_model[idx] = hestonmodel(
            S, Kvec[idx], r, float(T), kappa, theta, v0, rho, st, paths, xi, Z1, Z2
        )
    return C_model


def residualvector_surface(
    S,
    Tvec,
    Kvec,
    C_mkt,
    r,
    kappa,
    theta,
    v0,
    xi,
    rho,
    paths,
    Z_by_T,
    relative=True,
    eps=1e-3,
):
    C_model = model_prices_surface(
        S, Tvec, Kvec, r, kappa, theta, v0, xi, rho, paths, Z_by_T
    )
    res = C_mkt - C_model
    if relative:
        res = res / np.maximum(np.abs(C_mkt), eps)
    return res


def jacobian_surface_5(
    S,
    Tvec,
    Kvec,
    C_mkt,
    r,
    kappa,
    theta,
    v0,
    xi,
    rho,
    paths,
    Z_by_T,
    hkappa,
    htheta,
    hv0,
    hxi,
    hrho,
    relative=True,
):
    r0 = residualvector_surface(
        S,
        Tvec,
        Kvec,
        C_mkt,
        r,
        kappa,
        theta,
        v0,
        xi,
        rho,
        paths,
        Z_by_T,
        relative=relative,
    )
    r_k = residualvector_surface(
        S,
        Tvec,
        Kvec,
        C_mkt,
        r,
        kappa + hkappa,
        theta,
        v0,
        xi,
        rho,
        paths,
        Z_by_T,
        relative=relative,
    )
    dkappa = (r_k - r0) / hkappa
    r_th = residualvector_surface(
        S,
        Tvec,
        Kvec,
        C_mkt,
        r,
        kappa,
        theta + htheta,
        v0,
        xi,
        rho,
        paths,
        Z_by_T,
        relative=relative,
    )
    dtheta = (r_th - r0) / htheta
    r_v0 = residualvector_surface(
        S,
        Tvec,
        Kvec,
        C_mkt,
        r,
        kappa,
        theta,
        v0 + hv0,
        xi,
        rho,
        paths,
        Z_by_T,
        relative=relative,
    )
    dv0 = (r_v0 - r0) / hv0
    r_xi = residualvector_surface(
        S,
        Tvec,
        Kvec,
        C_mkt,
        r,
        kappa,
        theta,
        v0,
        xi + hxi,
        rho,
        paths,
        Z_by_T,
        relative=relative,
    )
    dxi = (r_xi - r0) / hxi
    rho2 = float(np.clip(rho + hrho, -0.9999, 0.9999))
    r_rho = residualvector_surface(
        S,
        Tvec,
        Kvec,
        C_mkt,
        r,
        kappa,
        theta,
        v0,
        xi,
        rho2,
        paths,
        Z_by_T,
        relative=relative,
    )
    drho = (r_rho - r0) / (rho2 - rho)
    J = np.column_stack([dkappa, dtheta, dv0, dxi, drho])
    return r0, J


def calibrate_heston_5(
    S,
    Tvec,
    Kvec,
    C_mkt,
    r,
    kappa0,
    theta0,
    v00,
    xi0,
    rho0,
    paths,
    Z_by_T,
    hkappa=1e-2,
    htheta=1e-3,
    hv0=1e-3,
    hxi=1e-2,
    hrho=1e-2,
    lam0=1e-2,
    lamup=10.0,
    lamdown=0.2,
    maxiter=40,
    tau=0.2,
    taurho=0.1,
    kappa_bounds=(1e-4, 20.0),
    theta_bounds=(1e-6, 2.0),
    v0_bounds=(1e-6, 2.0),
    xi_bounds=(1e-4, 5.0),
    rho_bounds=(-0.9999, 0.9999),
    relative=True,
    print_every=1,
):
    kappa = float(kappa0)
    theta = float(theta0)
    v0 = float(v00)
    xi = float(xi0)
    rho = float(rho0)
    lam = float(lam0)

    for it in range(1, maxiter + 1):
        r0, J = jacobian_surface_5(
            S,
            Tvec,
            Kvec,
            C_mkt,
            r,
            kappa,
            theta,
            v0,
            xi,
            rho,
            paths,
            Z_by_T,
            hkappa,
            htheta,
            hv0,
            hxi,
            hrho,
            relative=relative,
        )
        olderr = float(np.sum(r0**2))

        JTJ = J.T @ J
        JTr = J.T @ r0

        # Solve (J^T J + lam I) delta = J^T r
        # Then take a DESCENT step: params_new = params - delta
        delta = np.linalg.solve(JTJ + lam * np.eye(5), JTr)

        # step limiting (prevents wild jumps)
        max_kappa = tau * max(abs(kappa), 1e-6)
        max_theta = tau * max(abs(theta), 1e-6)
        max_v0 = tau * max(abs(v0), 1e-6)
        max_xi = tau * max(abs(xi), 1e-6)
        max_rho = taurho

        scale = 1.0
        scale = min(scale, max_kappa / max(abs(delta[0]), 1e-12))
        scale = min(scale, max_theta / max(abs(delta[1]), 1e-12))
        scale = min(scale, max_v0 / max(abs(delta[2]), 1e-12))
        scale = min(scale, max_xi / max(abs(delta[3]), 1e-12))
        scale = min(scale, max_rho / max(abs(delta[4]), 1e-12))
        delta = scale * delta

        k_try = float(np.clip(kappa - delta[0], kappa_bounds[0], kappa_bounds[1]))
        th_try = float(np.clip(theta - delta[1], theta_bounds[0], theta_bounds[1]))
        v0_try = float(np.clip(v0 - delta[2], v0_bounds[0], v0_bounds[1]))
        xi_try = float(np.clip(xi - delta[3], xi_bounds[0], xi_bounds[1]))
        rho_try = float(np.clip(rho - delta[4], rho_bounds[0], rho_bounds[1]))

        r_try = residualvector_surface(
            S,
            Tvec,
            Kvec,
            C_mkt,
            r,
            k_try,
            th_try,
            v0_try,
            xi_try,
            rho_try,
            paths,
            Z_by_T,
            relative=relative,
        )
        newerr = float(np.sum(r_try**2))

        if newerr < olderr:
            # accept
            kappa, theta, v0, xi, rho = k_try, th_try, v0_try, xi_try, rho_try
            lam = max(lam * lamdown, 1e-12)
            status = "ACCEPT"
        else:
            # reject
            lam = lam * lamup
            status = "REJECT"

        if (it % print_every == 0) or (it <= 10):
            print(
                f"iter {it:03d} {status} old={olderr:.6f} new={newerr:.6f} lam={lam:.2e} scale={scale:.3f} | "
                f"kappa={kappa:.6f} theta={theta:.6f} v0={v0:.6f} xi={xi:.6f} rho={rho:.6f}"
            )

    return kappa, theta, v0, xi, rho, lam


# input

So = 264.58
S = So
r = 0.08

Tvec = np.array(
    [
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        1 / 12,
        1 / 12,
        1 / 12,
        1 / 12,
        5 / 12,
        5 / 12,
        5 / 12,
        5 / 12,
    ],
    dtype=float,
)

Kvec = np.array(
    [
        230,
        235,
        240,
        245,
        250,
        255,
        260,
        275,
        280,
        285,
        290,
        230,
        260,
        275,
        290,
        230,
        260,
        275,
        290,
    ],
    dtype=float,
)

C_mkt = np.array(
    [
        45.8,
        39.65,
        37.27,
        34.95,
        31.78,
        28.18,
        25.55,
        17.05,
        15.20,
        12.69,
        11.35,
        38.14,
        11.85,
        3.64,
        0.66,
        43.90,
        23.00,
        15.80,
        9.53,
    ],
    dtype=float,
)

paths = 5000

Z_by_T = make_Z_by_T(Tvec, paths=paths, seed=134)
ivs = np.array(
    [
        bisection_iv(S, Kvec[i], r, Tvec[i], 1e-6, 3.0, C_mkt[i])
        for i in range(len(C_mkt))
    ],
    dtype=float,
)

T_max = float(np.max(Tvec))
idx_Tmax = np.where(Tvec == T_max)[0]
idx_atm = idx_Tmax[np.argmin(np.abs(Kvec[idx_Tmax] - S))]
iv_atm = ivs[idx_atm]
if not np.isfinite(iv_atm):
    iv_atm = 0.25

v0_init = float(iv_atm**2)
theta_init = float(iv_atm**2)
print("Init ATM iv used:", iv_atm, " => v0_init=theta_init=", v0_init)
kappa_init = 1.0
xi_init = 0.5
rho_init = -0.5
res0 = residualvector_surface(
    So,
    Tvec,
    Kvec,
    C_mkt,
    r,
    kappa_init,
    theta_init,
    v0_init,
    xi_init,
    rho_init,
    paths,
    Z_by_T,
    relative=True,
)
SSE_initial = float(np.sum(res0**2))
RMSE_initial = float(np.sqrt(np.mean(res0**2)))

print("\nInitial relative SSE:", SSE_initial)
print("Initial relative RMSE:", RMSE_initial)
kappa_fit, theta_fit, v0_fit, xi_fit, rho_fit, lam_fit = calibrate_heston_5(
    S=So,
    Tvec=Tvec,
    Kvec=Kvec,
    C_mkt=C_mkt,
    r=r,
    kappa0=kappa_init,
    theta0=theta_init,
    v00=v0_init,
    xi0=xi_init,
    rho0=rho_init,
    paths=paths,
    Z_by_T=Z_by_T,
    hkappa=1e-2,
    htheta=2e-3,
    hv0=2e-3,
    hxi=1e-2,
    hrho=1e-2,
    lam0=1e-2,
    lamup=10.0,
    lamdown=0.2,
    maxiter=40,
    tau=0.25,
    taurho=0.10,
    kappa_bounds=(1e-4, 20.0),
    theta_bounds=(1e-6, 2.0),
    v0_bounds=(1e-6, 2.0),
    xi_bounds=(1e-4, 5.0),
    rho_bounds=(-0.9999, 0.9999),
    relative=True,
    print_every=1,
)
print("\nFinal Stuff")
print("kappa =", kappa_fit)
print("theta =", theta_fit)
print("v0    =", v0_fit)
print("xi    =", xi_fit)
print("rho   =", rho_fit)
print("lam   =", lam_fit)
res_final = residualvector_surface(
    So,
    Tvec,
    Kvec,
    C_mkt,
    r,
    kappa_fit,
    theta_fit,
    v0_fit,
    xi_fit,
    rho_fit,
    paths,
    Z_by_T,
    relative=True,
)
print("Final SSE (relative):", float(np.sum(res_final**2)))
print("Final RMSE (relative):", float(np.sqrt(np.mean(res_final**2))))
