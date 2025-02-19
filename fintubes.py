#!/usr/bin/env python3
"""
Regression-builder and optimiser for finned tube refrigerant heat exchangers.

General structure:

Preamble
Import and Cleaning
Seed generation
Old Linear and Polynomial Regressors
scikit-learn regressors
keras regressors
Cavallini
optimisers
plotting routines
summary
entry

General TODO:

-Replace predictor() with builtin scipy Bounds() object
-Add Adam as an optimiser (modify sklearn? or bully tensorflow?)
-Get global DEGREE sweep to work
-remove older regressors?

"""

from functools import partial
from itertools import product
from markdown import markdown
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.constants import g, pi
from scipy.optimize import Bounds, minimize
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import sys

# keras utilities are SLOW to import, so we're only doing so if necessary.
if "--nn" in sys.argv:
    from fintubesnn import *

###################################################################################################
# Preamble.

help_message = """
Usage: fintubes [OPTION]

exploration of optimal parameters for finned-tube refrigerant heat exchangers.

-h, --help      prints this message
-d              drop CFC11 data
-c, --comp      compares models
--nn            trains a neural net
-o, --opt [M]   generates a model M and optimises around it.
                    Current options: "FC", "FC_PCA", "Cav", "PCA", "alphabeta", "nn"
                    "nn" will load a previously-generated neural net model
-p, --plot      generates plots
-pa, --plotall  generates more plots (standalone)
--publish       publish markdown file to HTML
-s, --seed      remake seed values
-t, --test      runs various t-tests
"""

# lists of variables for use in filtering.
# fmt: off
geo_vars = ["di", "hal", "alpha", "beta", "L"]
op_vars = ["Ts", "TsTp", "G"]
refrig_vars = ["DL", "DV", "RMUL", "RMUV", "CPL", "CPV", "CTL", "CTV", "TES", "RR", "PRED"]
categoricals = ["Author", "NG", "NFL", "Ptype"]
#FC_vars = ["Ts", "G", "x", "di", "beta", "CTL", "PRED"]
FC_vars = ["Ts", "G", "x", "di", "hal", "NG", "CTL", "PRED"]
Cav_vars = ["RMUL", "RMUV", "DL", "DV", "CPL", "CPV", "CTL", "RR", 
            "TsTp", "G", "x", "NG", "di", "alpha", "beta", "hal"]
PCA_vars = ["Ts", "TsTp", "G", "x", "di", "nf", "hal", "alpha", "beta", "L", 
            "DL", "DV", "RMUL", "RMUV", "CPL", "CPV", "CTL", "RR", "TES", "CTV", "PRED"]

#mapping of refrigerant to P-type. Rough and ready approximation of PRED.
pressure_type = {
    "11": "low",
    "113": "low",
    "123": "low",
    "114": "medium",
    "12": "high",
    "22": "high",
    "134": "high",
    "410": "high",
    "502": "high"
}
# fmt: on

###################################################################################################
# Import and cleaning.

fintubes = pd.read_csv(
    "globali.dat",
    delim_whitespace=True,
    dtype={"Author": "category", "NFL": "category", "NG": "category"},
)
fintubes["Ptype"] = pd.Categorical(
    fintubes["NFL"].map(lambda x: pressure_type.get(x, "unknown"))
)
fintubes.TsTp.replace(99.9, np.nan)

###################################################################################################
# seed finding routines

# cached values only from make_seed
# commented-out is 95th-percentile, left in is median
# fmt: off
alpha_beta_seed = [30.0, 10.0]

#           Ts    G      x     di      beta  CTL    PRED
#FC_seed =  [39.0, 440.0, 0.85, 0.0080, 12.0, 0.087, 0.25]
FC_seed = [39.0, 400.0, 0.62, 0.008, 15.0, 0.096, 0.33]

# "RMUL", "RMUV", "DL", "DV", "CPL", "CPV", "CTL", "RR", "TsTp", "G", "x", "NG", "di", "alpha", "beta", "hal"]
# Cav_seed = [0.000165, 0.000013, 1093.872120, 55.203041, 1777.032719, 
#             1402.843825, 0.115568, 253650.0, 43.687346, 466.617512,
#             0.788654, 1, 0.008108, 36.800000, 10.622120, 0.000216]
Cav_seed = [0.00014, 0.000014, 11100.0, 69.0, 1700.0, 
            1400.0, 0.095, 210000.0, 31.0, 400.0,
            0.62, 1, 0.008, 42.0, 15.0, 0.0002]

#Ts, TsTp, G, x, di, nf, hal, alpha, beta, L, DL, DV, RMUL, RMUV, CPL, CPV, CTL, RR, TES, CTV, PRED
PCA_seed = [39.4, 31.0, 400.0, 0.62, 0.008, 
            61.0, 0.0002, 43.0, 15.0, 2.1, 
            1100.0, 69.0, 0.00014, 0.000014, 1700.0,
            1400.0, 0.095, 210000.0, 0.0062, 0.016, 
            0.34]
# fmt: on


def make_low_P_seed(fintubes):
    """
    find the seed for low-pressure refrigerants. Does not drop Nozu, as that's a significant fraction of
    remaining data points (and the high performance is possibly relevant.)
    """
    max_h = fintubes[(fintubes.Ptype == "low")].h.quantile(0.5)
    upper_q = fintubes[(fintubes.Ptype == "low") & (fintubes.h > max_h)]
    return pd.Series(
        {
            "Ts": upper_q.Ts.mean(),
            "G": upper_q.G.mean(),
            "x": upper_q.x.mean(),
            "di": upper_q.di.mean(),
            "beta": upper_q.beta.mean(),
            "CTL": upper_q.CTL.mean(),
            "PRED": upper_q.PRED.mean(),
        }
    )


def make_FC_seed(fintubes):
    """
    makes the seed for the FC vars:
    saturation T, mass flow, vapour quality, diameter, beta, conductivity of liquid, reduced Pressure.
    """
    max_h = fintubes.h.quantile(0.5)
    upper_q = fintubes[(fintubes.h > max_h)][FC_vars].mean()
    return upper_q


def make_Cav_seed(fintubes):
    """
    makes the seed for Cavallini vars
    """
    max_h = fintubes.h.quantile(0.5)
    upper_q = fintubes[(fintubes.h > max_h)]
    seed = {}
    for c in Cav_vars:
        if c == "NG":
            seed.update({c: upper_q[c].mode()})
        else:
            seed.update({c: upper_q[c].mean()})
    return pd.Series(seed)


def make_noNoz_seed(fintubes):
    """
    makes seed. Drops Nozu. Unused.
    """
    max_h = fintubes[(fintubes.Author != "NOZ")].h.quantile(0.95)
    upper_q = fintubes[(fintubes.Author != "NOZ") & (fintubes.h > max_h)]
    return pd.Series(
        {
            "di": upper_q.di.mean(),
            "hal": upper_q.hal.mean(),
            "alpha": upper_q.alpha.mean(),
            "beta": upper_q.beta.mean(),
            "L": upper_q.L.mean(),
            "Ts": upper_q.Ts.mean(),
            "TsTp": upper_q.TsTp.mean(),
            "G": upper_q.G.mean(),
        }
    )


def make_PCA_seed(fintubes):
    """
    makes a 50% seed for PCA.
    """
    max_h = fintubes.h.quantile(0.5)
    upper_q = fintubes[fintubes.h > max_h][PCA_vars].mean()
    return upper_q


def make_seeds(fintubes):
    print(f"FC seed = {make_FC_seed(fintubes)}")
    print(f"Cav seed = {make_Cav_seed(fintubes)}")
    print(f"PCA seed = {make_PCA_seed(fintubes)}")


###################################################################################################
# t-tests


def fin_test(fintubes):
    print(
        f"""means for NG 1, 2 are {fintubes[(fintubes.NG == "1")].h.mean()}, {fintubes[(fintubes.NG == "2")].h.mean()},
              P-value is {ttest_ind(fintubes[(fintubes.NG == "1")].h, 
                                    fintubes[(fintubes.NG == "2")].h)[1]}
           """
    )


def ptype_test(fintubes):
    print(
        f"""means for low, high P are {fintubes[(fintubes.Ptype == "low")].h.mean()}, {fintubes[(fintubes.Ptype == "high")].h.mean()},
              P-value is {ttest_ind(fintubes[(fintubes.Ptype == "low")].h, 
                                    fintubes[(fintubes.Ptype == "high")].h)}
           """
    )


def test_all(fintubes):
    fin_test(fintubes)
    ptype_test(fintubes)


###################################################################################################
# Linear Regressors


def regress_linear_geometry(fintubes):
    model = sm.OLS(fintubes.h, fintubes[geo_vars]).fit()
    # with open("fintubes_geo_linear_regression.txt", "w") as f:
    #    f.write(repr(model.summary()))
    print(model.summary())


def regress_category_refrigerant(fintubes):
    model = smf.ols("h ~ Author + NG + NFL + Ptype", data=fintubes).fit()
    print(sm.stats.anova_lm(model, typ=2))


def regress_linear_refrigerant(fintubes):
    model = sm.OLS(fintubes.h, fintubes[refrig_vars]).fit()
    print(model.summary())


def regress_linear_all(fintubes):
    model = sm.OLS(fintubes.h, fintubes.drop(["h"] + categoricals, axis=1)).fit()
    # with open("fintubes_geo_linear_regression.txt", "w") as f:
    #    f.write(repr(model.summary()))
    print(model.summary())


###################################################################################################
# Older polynomial and PCA regressors

# default degree
DEGREE = 3

# default regularisation strength
ALPHA = 1.0


def regress_poly_geo(fintubes):
    """
    Older regressor for all noncategorical geometric features.
    """
    p = PolynomialFeatures(degree=DEGREE)
    xp = p.fit_transform(fintubes[geo_vars])
    model = sm.OLS(fintubes.h, xp).fit()
    plt.plot(model.predict(xp))
    return model


def regress_poly_alpha_beta(fintubes):
    """
    Older regressor for just alpha and beta.
    """
    p = PolynomialFeatures(degree=DEGREE)
    xp = p.fit_transform(fintubes[["alpha", "beta"]])
    model = sm.OLS(fintubes.h, xp).fit()
    return model


def regress_poly_sweep(fintubes):
    """
    Test performance of poly fit with varying degree. Unused.
    """

    def regress_poly_degree(fintubes, degree):
        global DEGREE
        DEGREE = degree
        return regress_poly_geo(fintubes)

    results = [
        (regress_poly_degree(fintubes, degree).rsquared) for degree in range(1, 5)
    ]
    plt.plot([r[1] for r in results])
    plt.xlabel("degree")
    plt.ylabel("R-squared")
    plt.title("R-squared against polynomial degree for geometric parameters")
    plt.show()


def regress_individual(fintubes):
    regress_category_refrigerant(fintubes)
    regress_poly_sweep(fintubes)
    regress_linear_all(fintubes)
    regress_linear_geometry(fintubes)
    regress_linear_refrigerant(fintubes)


###################################################################################################
# Newer polynomial and PCA regressors


def regress_FC_poly(fintubes, plot):
    """
    First scikit-learn implementation with validation of regularisation strength. Defauls to degree 3.
    polynomial features with 7 inputs:

    NOTE: regress_FC_kernel should do the same thing, but slightly more elegantly.

    degree: 1, 2,  3,   4,   5,   6,    7
    dof:    8, 36, 120, 330, 792, 1716, 3432
    """
    alphas = np.logspace(-6, 6, 13)
    print(
        f"degrees of freedom: {PolynomialFeatures(DEGREE).fit_transform(fintubes[FC_vars]).shape[1]}"
    )
    x_train, x_test, y_train, y_test = train_test_split(
        fintubes[FC_vars], fintubes.h, train_size=0.6
    )
    p = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("make_poly", PolynomialFeatures(DEGREE)),
            ("ridge", RidgeCV(alphas=alphas, scoring="r2")),
        ]
    )
    p.fit(x_train, y_train)
    return p


def regress_all_PCA(fintubes, plot):
    """
    PCA-based polynomial regressor.
    """
    # drop categorials, target var, and modelled var (that has some NaNs)
    x_train, x_test, y_train, y_test = train_test_split(
        fintubes[PCA_vars], fintubes.h, train_size=0.6
    )
    p = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=6)),
            ("ridge", KernelRidge(alpha=ALPHA, kernel="poly")),
        ]
    )
    p.fit(x_train, y_train)
    if plot:
        print(f"score for PCA / all vars, kernel ridge: {p.score(x_test, y_test):.2f}")
    return p


def regress_FC_PCA(fintubes, plot):
    """
    PCA-based polynomial regressor.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        fintubes[FC_vars], fintubes.h, train_size=0.6
    )
    p = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=6)),
            ("ridge", KernelRidge(alpha=ALPHA, kernel="poly")),
        ]
    )
    p.fit(x_train, y_train)
    if plot:
        plot_parity(p, x_train, x_test, y_train, y_test)
        print(f"score for PCA / FC vars, kernel ridge: {p.score(x_test, y_test):.2f}")
    return p


def regress_FC_kernel(fintubes, plot):
    """
    Another polynomial regressor, using the inbuild polynomial kernel.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        fintubes[FC_vars], fintubes.h, train_size=0.6
    )
    p = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", KernelRidge(alpha=ALPHA, kernel="poly")),
        ]
    )
    p.fit(x_train, y_train)
    if plot:
        plot_parity(p, x_train, x_test, y_train, y_test)
        print(f"score for FC vars, kernel ridge: {p.score(x_test, y_test):.2f}")
    return p


def PCA_explained(fintubes):
    """
    Proportion of variance explained by each degree of PCA.
    """
    to_drop = categoricals + ["h"]
    pca = PCA()
    pca.fit(fintubes.drop(to_drop, axis=1))
    plt.plot(pca.explained_variance_ratio_)
    plt.show()


###################################################################################################
# Cavallini utility functions


def Cavallini(row):
    """
    Implements Cavallini (2009)
    """
    # testcase iloc[4123]
    n_rat = lambda row: (4064.4 * row.di + 23.257) / row.nf
    C = lambda row: 1.0 if n_rat(row) >= 0.8 else n_rat(row) ** 1.904
    C1 = lambda row: 1.0 if J_g(row) >= J_g_crit(row) else J_g(row) / J_g_crit(row)
    Fr = lambda row: row.G**2 / (g * row.di * (row.DL - row.DV) ** 2)
    X_tt = (
        lambda row: (row.RMUL / row.RMUV) ** 0.1
        * (row.DV / row.DL) ** 0.5
        * ((1 - row.x) / row.x) ** 0.9
    )
    J_g = lambda row: row.x * row.G / (g * row.di * row.DV * (row.DL - row.DV)) ** 0.5
    J_g_crit = (
        lambda row: 0.6
        * ((7.5 / (4.3 * X_tt(row) ** 1.1111 + 1)) ** -3 + 2.5**-3) ** -0.3333
    )
    Rx = lambda row: (
        (
            (
                2
                * row.hal
                * row.nf
                * ((1 - np.sin(np.radians(row.alpha)) / 2))
                / (pi * row.di * np.cos(np.radians(row.alpha) / 2))
            )
            + 1.0
        )
        / np.cos(np.radians(row.beta))
    )
    Prandtl_l = lambda row: row.RMUL * row.CPL / row.CTL
    h_lo = (
        lambda row: 0.023
        * (row.CTL / row.di)
        * (row.G * row.di / row.RMUL) ** 0.8
        * Prandtl_l(row) ** 0.4
    )
    A = lambda row: 1.0 + 1.119 * Fr(row) ** -0.3821 * (Rx(row) - 1) ** 0.3586
    h_as = lambda row: h_lo(row) * (
        1
        + (
            1.128
            * row.x**0.817
            * (row.DL / row.DV) ** 0.3685
            * (row.RMUL / row.RMUV) ** 0.2363
            * (1 - row.RMUV / row.RMUL) ** 2.144
            * Prandtl_l(row) ** -0.1
        )
    )
    h_ds = lambda row: (
        0.725
        / (1 + 0.741 * ((1 - row.x) / row.x) ** 0.3321)
        * (
            ((row.CTL**3) * row.DL * (row.DL - row.DV) * g * row.RR)
            / (row.RMUL * row.di * row.TsTp)
        )
        ** 0.25
    )
    h_a = lambda row: h_as(row) * A(row) * C(row)
    h_d = lambda row: (
        C(row)
        * (
            h_ds(row)
            * (2.4 * row.x**0.1206 * (Rx(row) - 1) ** 1.466 * C1(row) ** 0.6875 + 1)
            + h_lo(row) * (Rx(row) * (1 - row.x**0.087))
        )
    )
    try:
        assert row.x <= 1.0  # nonphysical vapour quality
        return (h_a(row) ** 3 + h_d(row) ** 3) ** 0.333
    except (ZeroDivisionError, AssertionError):
        return np.nan


def apply_Cavallini(fintubes):
    fintubes["h_Cavallini"] = fintubes.apply(Cavallini, axis=1)


def test_Cavallini():
    print(Cavallini(fintubes.iloc[4123]))


###################################################################################################
# optimisers


def optimise_alpha_beta(fintubes, model):
    """
    Optimises alpha and beta only.
    """

    def predictor(model, x, invert=False):
        """
        Returns an optimiser-friendly prediction given the state vector and model.

        CAUTION: minimise can't accept pd.Series unfortunately, so x needs to be in the form:
        [alpha, beta, other] in that order.
        """

        # x = np.array([[1.0,1.0]], np.float64)
        def banned(x):
            """
            Illegal regions for alpha-beta only.
            """
            return (
                (x[0] > 90.0)
                or (x[0] < 0.0)
                or (x[1] > 90.0)
                or (x[1] < 0.0)
                or (x[0] < (2.8 * x[1]) - 13.0 or (x[0] < (-2.3 * x[1]) + 30.0))
            )

        p = PolynomialFeatures(degree=DEGREE)
        xp2 = p.fit_transform([x])
        # print(f"{xp2.shape=}")
        if banned(x):
            return 0.0
        else:
            return (-1.0 if invert else 1.0) * model.predict(xp2)

    f = partial(predictor, model, invert=True)
    r = minimize(f, alpha_beta_seed, method="Powell", options={"maxiter": 1000})
    print(f"{r.x=}")
    print(f"optimal: {-1 * f(r.x)}")
    return -1 * f(r.x)


def optimise_sweep(fintubes):
    """
    Optimises with a sweep of polynomial degree. Unused.
    """
    x = []
    y = []
    global DEGREE
    for degree in range(2, 10):
        DEGREE = degree
        model = regress_poly_alpha_beta(fintubes)
        res = optimise_alpha_beta(fintubes, model)
        x += [degree]
        y += [res]
    plt.plot(y)
    plt.show()


def optimise_FC(fintubes, model):
    """
    Optimises the seven FC variables (Saturation temperature, mass flow rate, vapour quality, diameter,
    helix angle, Liquid Thermal Conductivity, Reduced Pressure), given a model.
    """

    # TODO: write up the effect of different optimisers.
    # Gradient-free seem to drastically outperform gradient-based.
    def predictor(model, x):
        """
        Applies the model given a parameter vector x, and returns the negative so that it's
        minimiser-friendly. Also coerces NaN-cases to 0 (i.e. high, so an optimiser will
        never optimise to NaN).
        """

        def banned(x):
            """
            Illegal regions for FC_vars
            """

            FC_limits = [
                (-15.0, 74.11),
                (21.0, 919.0),
                (0.0, 1.0),
                (0.00595, 0.01587),
                (0.0, 30.0),
                (0.04892, 0.51773),
                (0.027999999999999997, 0.674),
            ]
            return any((i < j[0]) or (i > j[1]) for i, j in zip(x, FC_limits))

        if banned(x):
            return 0.0
        if isinstance(model, Pipeline):
            p = model.predict([x])[0]
        else:  # must be keras
            p = model.predict([list(x)])[0]
        if p == np.nan:
            return 0.0
        return -1.0 * p

    f = partial(predictor, model)
    bounds = Bounds(
        [-15.0, 21.0, 0.0, 0.00595, 0.0, 0.04, 0.51773, 0.0279],
        [74.11, 919.0, 1.0, 0.01587, 30.0, 0.51773, 0.674],
    )
    r = minimize(f, FC_seed, method="Powell", options={"maxiter": 1000})
    print("r.x = ", [f"{i:2g}" for i in r.x])
    print(f"optimal: {-1 * f(r.x)}")


def optimise_Cav(fintubes):
    """
    Optimises the Cavallini variables, using Cavallini as objective function.
    """

    def predictor(x):
        def banned(x):
            """
            Illegal regions for Cav_vars
            """
            Cav_limits = [
                (fintubes[i].min(), fintubes[i].max()) for i in Cav_vars if i != "NG"
            ]
            return (
                (x[11] > 2)
                or (x[11] < 1)
                or any((i < j[0]) or (i > j[1]) for i, j in zip(x, Cav_limits))
            )

        row = pd.Series(dict(zip(Cav_vars, x)))
        h = Cavallini(row)
        if np.isnan(h) or np.isposinf(h) or np.isneginf(h):
            return 0.0
        return -1.0 * h

    f = partial(predictor)
    print(predictor(Cav_seed))
    b = Bounds(fintubes[Cav_vars].min(), fintubes[Cav_vars].max())
    print(b)
    r = minimize(f, Cav_seed, method="TNC", options={"maxiter": 1000}, bounds=b)
    print([f"{i:2g}" for i in r.x])
    print(f"optimal: {-1 * f(r.x)}")


def optimise_PCA(fintubes, model):
    """
    Optimises all PCA-friendly variables.
    """

    def predictor(model, x):
        def banned(x):
            """
            Illegal regions for PCA_vars
            """
            PCA_limits = [(fintubes[i].min(), fintubes[i].max()) for i in PCA_vars]
            return any((i < j[0]) or (i > j[1]) for i, j in zip(x, PCA_limits))

        if banned(x):
            y = 0.0

        else:
            y = -1.0 * model.predict([x])[0]
        # print(y)
        return y

    f = partial(predictor, model)
    r = minimize(f, PCA_seed, method="Powell", options={"maxiter": 1000})
    print([f"{i:2g}" for i in r.x])
    print(f"{r.x=}")
    print(f"optimal: {-1 * f(r.x)}")


###################################################################################################
# Plotting routines

z_max = 30_000.0


def plot_correlations(fintubes):
    """
    Prints a correlation matrix with all non-categorical factors.
    """
    corr = np.corrcoef(fintubes.drop(categoricals, axis=1), rowvar=False)
    smg.plot_corr(
        corr,
        xnames=fintubes.drop(categoricals, axis=1).columns,
        ynames=fintubes.drop(categoricals, axis=1).columns,
        normcolor=True,
    )
    plt.show()


def plot_reduced_correlations(fintubes):
    """
    Prints a correlation matrix, including only geometrical and operational parameters.
    """
    corr = np.corrcoef(fintubes[["h"] + geo_vars + op_vars], rowvar=False)
    smg.plot_corr(
        corr,
        xnames=["h"] + geo_vars + op_vars,
        ynames=["h"] + geo_vars + op_vars,
        normcolor=True,
    )
    plt.title("Correlation matrix (Geometric and Operational parameters)")
    plt.show()


def plot_diameter(fintubes):
    plt.scatter(fintubes["di"], fintubes["h"], c="black", marker="x")
    plt.xlabel("diameter")
    plt.ylabel("h")
    plt.show()


def plot_hal(fintubes):
    plt.scatter(fintubes["hal"], fintubes["h"], c="black", marker="x")
    plt.xlabel("hal")
    plt.ylabel("h")
    plt.show()


def plot_alpha(fintubes):
    plt.scatter(fintubes["alpha"], fintubes["h"], c="black", marker="x")
    plt.xlabel("alpha")
    plt.ylabel("h")
    plt.show()


def plot_alpha_beta_model(fintubes, model):

    def predictor(model, x, invert=False):
        """
        Returns an optimiser-friendly prediction given the state vector and model.

        CAUTION: minimise can't accept pd.Series unfortunately, so x needs to be in the form:
        [alpha, beta, other] in that order.
        """

        def banned(x):
            return (
                (x[0] > 90.0)
                or (x[0] < 0.0)
                or (x[1] > 90.0)
                or (x[1] < 0.0)
                or (x[0] < (2.8 * x[1]) - 13.0 or (x[0] < (-2.3 * x[1]) + 30.0))
            )

        p = PolynomialFeatures(degree=DEGREE)
        xp2 = p.fit_transform([x])
        if banned(x):
            return 0.0
        else:
            return (-1.0 if invert else 1.0) * model.predict(xp2)

    fintubes["h_pred"] = fintubes.apply(
        lambda row: predictor(model, [row.alpha, row.beta]), axis=1
    )
    results = [fintubes.alpha, fintubes.beta, fintubes.h_pred]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(
        fintubes.alpha, fintubes.beta, fintubes.h_pred, cmap=cm.plasma, linewidth=0.1
    )
    ax.set_xlabel("alpha (°)")
    ax.set_ylabel("beta (°)")
    ax.set_zlabel("h $_{predicted}$ (W.m$^{-2}$.K$^{-1}$)")
    ax.set_zlim3d(0.0, z_max)
    plt.show()


def plot_alpha_beta_FC(fintubes, model):
    fintubes["h_pred"] = model.predict(fintubes[FC_vars])
    results = [fintubes.alpha, fintubes.beta, fintubes.h_pred]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(
        fintubes.alpha, fintubes.beta, fintubes.h_pred, cmap=cm.plasma, linewidth=0.1
    )
    ax.set_xlabel("alpha (°)")
    ax.set_ylabel("beta (°)")
    ax.set_zlabel("h $_{predicted}$ (W.m$^{-2}$.K$^{-1}$)")
    ax.set_zlim3d(0.0, z_max)
    plt.show()


def plot_alpha_beta_Cav(fintubes):
    if "h_Cavallini" not in fintubes.columns:
        apply_Cavallini(fintubes)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(
        fintubes.alpha,
        fintubes.beta,
        fintubes.h_Cavallini,
        cmap=cm.plasma,
        linewidth=0.1,
    )
    ax.set_xlabel("alpha (°)")
    ax.set_ylabel("beta (°)")
    ax.set_zlabel("h_{Cavallini} (W.m$^{-2}$.K$^{-1}$)")
    ax.set_zlim(0.0, z_max)
    plt.show()


def plot_all_geo_model(fintubes, model):
    di = [fintubes.di.mean()] * 50
    hal = [fintubes.hal.mean()] * 50
    L = [fintubes.L.mean()] * 50
    p = PolynomialFeatures(degree=DEGREE)
    a = np.linspace(fintubes.alpha.min(), fintubes.alpha.max())
    b = np.linspace(fintubes.beta.min(), fintubes.beta.max())
    c = np.linspace(fintubes.hal.min(), fintubes.hal.max())
    results = [
        (x, y, model.predict(p.fit_transform([[x, y, fintubes.hal.mean()]]))[0])
        for x, y in product(a, b)
    ]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(
        [i[0] for i in results],
        [i[1] for i in results],
        [np.clip(i[2], 0.0, z_max) for i in results],
        cmap=cm.plasma,
        linewidth=0.1,
    )
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_zlabel("h (predicted) (W.m$^{-2}$.K$^{-1}$)")
    ax.set_zlim3d(0.0, z_max)
    plt.show()


def plot_alpha_beta_surface(fintubes):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(
        fintubes.alpha, fintubes.beta, fintubes.h, cmap=cm.plasma, linewidth=0.1
    )
    ax.set_xlabel("alpha (°)")
    ax.set_ylabel("beta (°)")
    ax.set_zlabel("h (W.m$^{-2}$.K$^{-1}$)")
    ax.set_zlim(0.0, z_max)
    plt.show()


def plot_di_hal_surface(fintubes):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(
        fintubes.di, fintubes.hal, fintubes.h, cmap=cm.plasma, linewidth=0.1
    )
    ax.set_xlabel("diameter (m)")
    ax.set_ylabel("hal (m)")
    ax.set_zlabel("h (W.m$^{-2}$.K$^{-1}$)")
    ax.set_zlim(0.0, 20000.0)
    plt.show()


def violins(fintubes):
    """
    Makes a violin plot.
    """
    ax = sns.violinplot(
        y=fintubes["h"],
        x=fintubes["NFL"],
        hue=fintubes["NG"],
        # title="violin plot for h",
        split=True,
    )
    ax.set_xlabel("Refrigerant type")
    ax.set_ylabel("h (W.m$^{-2}$.K$^{-1}$)")
    ax.set_title("violin plot for h")
    ax.legend(labels=["Microfin", "Crossgrooved"])
    plt.show()


def plot_alpha_beta_surface_plus_cav(fintubes):
    """
    Side-by-side plot of real and Cav-modelled data. We could probably abstract this.
    """
    if "h_Cavallini" not in fintubes.columns:
        apply_Cavallini(fintubes)
    fig = plt.figure()
    G = gridspec.GridSpec(3, 2)
    ax1 = fig.add_subplot(G[0:2, 0], projection="3d")
    ax2 = fig.add_subplot(G[0:2, 1], projection="3d")
    ax1.plot_trisurf(
        fintubes.alpha, fintubes.beta, fintubes.h, cmap=cm.plasma, linewidth=0.1
    )
    ax2.plot_trisurf(
        fintubes.alpha,
        fintubes.beta,
        fintubes.h_Cavallini,
        cmap=cm.plasma,
        linewidth=0.1,
    )
    ax1.set_xlabel("alpha (°)")
    ax1.set_ylabel("beta (°)")
    ax1.set_zlabel("h (W.m$^{-2}$.K$^{-1}$)")
    ax2.set_xlabel("alpha (°)")
    ax2.set_ylabel("beta (°)")
    ax2.set_zlabel("h$_{Cavallini}$ (W.m$^{-2}$.K$^{-1}$)")
    ax1.set_zlim(0.0, z_max)
    ax2.set_zlim(0.0, z_max)
    plt.show()


def plot_parity(model, x_train, x_test, y_train, y_test):
    """
    Generates a parity plot, with training/test data colour coded.
    """

    def predictor(row, vars=FC_vars):
        if isinstance(model, Pipeline):
            return model.predict([row[vars]])[0]
        else:  # must be keras, so needs a slightly different input format.
            return model.predict([list(row[vars])])[0]

    fintubes_test = x_test.join(y_test)
    fintubes_train = x_train.join(y_train)
    fintubes_test["h_pred"] = fintubes_test.apply(predictor, axis=1)
    fintubes_test = fintubes_test.sort_values(["h"])
    fintubes_train["h_pred"] = fintubes_train.apply(predictor, axis=1)
    fintubes_train = fintubes_train.sort_values(["h"])
    fig = plt.figure()
    plt.scatter(
        fintubes_train.h,
        fintubes_train.h_pred,
        marker="+",
        color="black",
        label="train",
    )
    plt.scatter(
        fintubes_test.h, fintubes_test.h_pred, marker="+", color="red", label="test"
    )
    plt.plot([0, 50000], [0, 50000], linewidth=1, linestyle="dashed", color="black")
    plt.legend()
    plt.xlabel("h$_{exp}$ (W.m$^{-2}$.K$^{-1}$)")
    plt.ylabel("h$_{pred}$ (W.m$^{-2}$.K$^{-1}$)")
    plt.show()


def plot_parity_Cav(fintubes):
    """
    Generates a parity plot for Cavallini function.
    """
    fintubes["h_pred"] = fintubes.apply(Cavallini, axis=1)
    fintubes = fintubes.sort_values(["h"])
    fig = plt.figure()
    plt.scatter(fintubes.h, fintubes.h_pred, marker="+", color="black")
    plt.plot([0, 50000], [0, 50000], linewidth=1, linestyle="dashed", color="black")
    plt.xlabel("h$_{exp}$ (W.m$^{-2}$.K$^{-1}$)")
    plt.ylabel("h$_{pred}$ (W.m$^{-2}$.K$^{-1}$)")
    plt.show()


def plot_all(fintubes):
    plot_correlations(fintubes)
    violins(fintubes)
    plot_di_hal_surface(fintubes)
    plot_alpha_beta_surface(fintubes)
    plot_diameter(fintubes)
    plot_alpha(fintubes)
    plot_hal(fintubes)
    apply_Cavallini(fintubes)
    plot_alpha_beta_Cav(fintubes)


###################################################################################################
# Model comparison


def compare_models(fintubes):
    apply_Cavallini(fintubes)
    print(
        f"Cavallini R-squared = {sm.OLS(fintubes.h, fintubes.h_Cavallini, missing='drop').fit().rsquared:.2f}"
    )
    x_test, y_test, model = regress_FC_poly(fintubes, False)
    print(
        f"alpha auto-fit to {model.named_steps['ridge'].alpha_} when degree = {DEGREE}"
    )
    print(f"score for FC vars, poly ridge: {model.score(x_test, y_test):.2f}")
    regress_FC_kernel(fintubes, True)
    regress_FC_PCA(fintubes, True)
    regress_all_PCA(fintubes, True)


###################################################################################################
if __name__ == "__main__":
    plot = False
    if "-h" in sys.argv or "--help" in sys.argv:
        print(help_message)
    if "-d" in sys.argv:
        fintubes = fintubes[fintubes.NFL != "11"]
    if "-p" in sys.argv or "--plot" in sys.argv:
        plot = True
    if "-pa" in sys.argv or "--plotall" in sys.argv:
        plot_all(fintubes)
    if "--Cav" in sys.argv:
        apply_Cavallini(fintubes)
        print(
            f"Cavallini R-squared = {sm.OLS(fintubes.h, fintubes.h_Cavallini, missing='drop').fit().rsquared:.2f}"
        )
        plot_parity_Cav(fintubes)
    if "-t" in sys.argv or "--test" in sys.argv:
        test_all(fintubes)
    if "-s" in sys.argv or "--seed" in sys.argv:
        make_seeds(fintubes)
    if "--nn" in sys.argv:
        if "--opt" not in sys.argv:
            model = regress_NN(fintubes, plot)
            fname = input(
                "what name would you like to give this model? (leave blank to cancel)\n"
            )
            if fname:
                model.save("models/" + fname)
    if "--opt" in sys.argv or "-o" in sys.argv:
        try:
            m_type = sys.argv[sys.argv.index("--opt") + 1]
            if m_type == "alphabeta":
                model = regress_poly_alpha_beta(fintubes)
                print(model.summary())
                plot_alpha_beta_model(fintubes, model)
                optimise_alpha_beta(fintubes, model)
            elif m_type == "Cav":
                plot_parity_Cav(fintubes)
                optimise_Cav(fintubes)
            elif m_type == "FC":
                model = regress_FC_kernel(fintubes, plot)
                optimise_FC(fintubes, model)
            elif m_type == "FC_PCA":
                model = regress_FC_PCA(fintubes, plot)
                optimise_FC(fintubes, model)
            elif m_type == "nn":
                from keras.models import load_model

                model = load_model("models/" + input("model name?"))
                optimise_FC(fintubes, model)
            elif m_type == "PCA":
                model = regress_all_PCA(fintubes, plot)
                optimise_PCA(fintubes, model)
            else:
                raise IndexError
        except IndexError:
            print("I don't know which model to optimise with. See help...")
            print(help_message)
    if ("-c" in sys.argv) or ("--comp" in sys.argv):
        compare_models(fintubes)
    if "--publish" in sys.argv:
        with open("README.md", "r") as f, open("README.html", "w") as g:
            g.write(
                markdown(
                    f.read(),
                    extensions=[
                        "tables",
                        "markdown.extensions.latex",
                        "markdown.extensions.imagetob64",
                    ],
                )
            )
