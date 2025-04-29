import numpy as np
from scipy.constants import Stefan_Boltzmann
import pandas as pd
from scipy import stats, optimize

S = 1370  # W/m^2
R = 6.38e6  # m


def ReadSiple2():
    # pd.read_csv("siple_1", usecols=(3, 4), names=["year", "CO2"])
    data_old = pd.read_csv("data/siple_2.csv", usecols=(1, 2), header=0, names=["year", "CO2"])
    data_old["stddev"] = 1.5
    return (data_old)


def read_atmospheric_CO2_data():
    data_new = pd.read_csv("data/Maunoa_Lao_atmospheric_CO2_levels_1956-2024.csv",
                           skiprows=44, usecols=(0, 1, 2), header=0, names=["year", "CO2", "stddev"])
    data_old = ReadSiple2()
    data = pd.concat([data_old, data_new], ignore_index=True)
    return data


def read_temperature_data():
    """Reads the temperature data from the file and returns it as a numpy array."""
    data: pd.DataFrame = pd.read_csv("data/globalTemperatureAnomoly1900-2024.csv",
                                     skiprows=4, usecols=(0, 1), header=0, names=["year", "anomaly"])
    data["temperature"] = data["anomaly"] + 286.9
    data["stddev"] = 0.025

    return data[["year", "temperature", "stddev"]]


def read_data():
    temp_data = read_temperature_data()
    co2_data = read_atmospheric_CO2_data()
    temp_data["year"] = pd.to_datetime(temp_data["year"], format="%Y")
    co2_data["year"] = pd.to_datetime(co2_data["year"], format="%Y")
    temp_data.set_index("year", inplace=True)
    co2_data.set_index("year", inplace=True)
    data = pd.merge(co2_data, temp_data, left_index=True, right_index=True)
    return data


def embedded_f_linear(f0, f1, Y_C02):
    return f0 + f1 * Y_C02


def model_T(alpha, f0, f1, Y_C02):
    """
    Returns the temperature T for a given alpha, f0, and f1.
    """
    f = embedded_f_linear(f0, f1, Y_C02)
    return np.power((S / 4) * (1 - alpha) / (Stefan_Boltzmann * (1 - 0.5 * f)), 0.25)


def model_T_simple(alpha, f):
    """
    Returns the temperature T for a given alpha and f.
    """
    return np.power((S / 4) * (1 - alpha) / (Stefan_Boltzmann * (1 - 0.5 * f)), 0.25)


def dmodel_dco2(alpha, f0, f1, Y_C02):
    """
    Returns the derivative of the model with respect to CO2.
    """
    f = embedded_f_linear(f0, f1, Y_C02)
    c = np.power((S / 4) * (1 - alpha) / (Stefan_Boltzmann), 0.25)
    return c * -0.25 * np.power(1 - f / 2, -1.25) * (-0.5 * f1)


def compute_uncertainty(data, alpha, f0, f1):
    dT_dCO2 = dmodel_dco2(alpha, f0, f1, data["CO2"].to_numpy())
    var_CO2 = (data["stddev_x"].to_numpy() ** 2) * (dT_dCO2 ** 2)
    var_T = (data["stddev_y"].to_numpy() ** 2) + var_CO2
    return var_T


def norm_logpdf(x, mean, var):
    """
    Returns the log of the normal probability density function evaluated at x, with mean and variance var.
    Omits the constant -0.5 * log(2 * pi).
    """
    return -0.5 * (np.log(var) + (x - mean) * (x - mean) / var)


def likelihood(alpha, f0, f1, data=None):
    """
    Returns the log of the likelihood function alpha, f0, f1
    """
    if alpha < 0 or alpha > 1:
        return -np.inf
    if f0 < 0 or f0 > 1:
        return -np.inf
    if data is None:
        data = CALIBRATION_DATA
    CO2 = data["CO2"].to_numpy()
    f = embedded_f_linear(f0, f1, CO2)
    if np.any(f < 0) or np.any(f > 1):
        return -np.inf
    var = compute_uncertainty(data, alpha, f0, f1)
    T = data["temperature"].to_numpy()
    T_m = model_T(alpha, f0, f1, CO2)
    return sum(norm_logpdf(T_m[i], T[i], var[i]) for i in range(len(T_m)))


NEG_INF = -1e10


def compute_map(data, alpha, f0, f1):
    """Compute the MAP estimate of the parameters."""
    # Compute the MAP estimate of the parameters
    def func(params):
        alpha, f0, f1 = params
        l = -max(likelihood(alpha, f0, f1, data), NEG_INF)
        if np.isnan(l):
            return 1e10
        return l
    CO2 = data["CO2"].to_numpy()
    A = np.column_stack([np.zeros_like(CO2), np.ones_like(CO2), CO2])
    lb = np.zeros_like(CO2)
    ub = np.ones_like(CO2)

    res = optimize.minimize(
        func,
        (alpha, f0, f1),
        bounds=optimize.Bounds([0, 0, -0.005], [1, 1, 0.005], [True, True, True]),
        # constraints=[optimize.LinearConstraint(A, lb, ub, keep_feasible=True)],
        # jac="3-point",
        # method="trust-constr",
        method="Nelder-Mead",
        tol=1e-16
    )
    if res.success:
        print("\tOptimization was successful: ", res.x)
    else:
        print("\tOptimization failed.")
    return res


def update_uncertainty(data, alpha, f0, f1):
    data["var_T"] = compute_uncertainty(data, alpha, f0, f1)
    return data


CALIBRATION_DATA = read_data()
INITIAL_PARAMS = compute_map(CALIBRATION_DATA, 0.3, 0.7, 0.0002).x
CALIBRATION_DATA = update_uncertainty(CALIBRATION_DATA, *INITIAL_PARAMS)
T = CALIBRATION_DATA["temperature"].to_numpy()
CO2 = CALIBRATION_DATA["CO2"].to_numpy()
VAR_T = CALIBRATION_DATA["var_T"].to_numpy()
print(VAR_T[:20])

const_term = np.sum(-0.5 * np.log(2 * np.pi * VAR_T))


PDFS = [stats.norm(loc=T[i], scale=np.sqrt(VAR_T[i])) for i in range(len(T))]
