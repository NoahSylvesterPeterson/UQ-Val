import numpy as np
from scipy.constants import Stefan_Boltzmann
import pandas as pd
from scipy import stats, optimize
from abc import ABC, abstractmethod

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


DATA = read_data()
CO2 = DATA["CO2"].to_numpy()
TEMPERATURE = DATA["temperature"].to_numpy()
STDDEV_CO2 = DATA["stddev_x"].to_numpy()
STDDEV_T = DATA["stddev_y"].to_numpy()


def norm_logpdf(x, mean, var):
    """
    Returns the log of the normal probability density function evaluated at x, with mean and variance var.
    Omits the constant -0.5 * log(2 * pi).
    """
    return -0.5 * (np.log(var) + (x - mean) ** 2 / var)


class Model(ABC):
    @abstractmethod
    def embedded_f(self, params, Y_C02):
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def df_dco2(self, alpha, params, Y_C02):
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    @abstractmethod
    def num_params(self):
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def num_data(self):
        return len(TEMPERATURE)

    def dmodel_dco2(self, params, Y_C02):
        """
        Returns the derivative of the model with respect to CO2.
        """
        f = self.embedded_f(params, Y_C02)
        c = np.power((S / 4) * (1 - params[0]) / (Stefan_Boltzmann), 0.25)
        return c * -0.25 * np.power(1 - 0.5 * f, -1.25) * (-0.5 * self.df_dco2(params, Y_C02))

    def variance(self, params):
        """
        Returns the variance in the temperature T for a given set of parameters.
        """
        return STDDEV_T ** 2 + STDDEV_CO2 ** 2 * np.power(self.dmodel_dco2(params, CO2), 2)

    def predict(self, params, CO2):
        """
        Returns the predicted temperature T for a given set of parameters.
        """
        return np.power((S / 4) * (1 - params[0]) / (Stefan_Boltzmann *
                        (1 - 0.5 * self.embedded_f(params, CO2))), 0.25)

    def evaluate(self, params):
        """
        Returns the temperature T for a given set of parameters.
        """
        return self.predict(params, CO2)

    def log_likelihood(self, params):
        """
        Returns the log likelihood.
        """
        f = self.embedded_f(params, CO2)
        if np.any(f < 0) or np.any(f > 1):
            return -np.inf
        return np.sum(norm_logpdf(self.evaluate(params), TEMPERATURE, self.variance(params)))


class ModelConstant(Model):
    def __init__(self):
        super().__init__()

    def embedded_f(self, params, Y_C02):
        return params[1]

    def df_dco2(alpha, params, Y_C02):
        """
        Returns the derivative of the model with respect to CO2.
        """
        return 0.0

    @property
    def num_params(self):
        return 2

    def __str__(self):
        return "ModelConstant"


class ModelLinear(Model):
    def __init__(self):
        super().__init__()

    def embedded_f(self, params, Y_C02):
        return params[1] + params[2] * Y_C02

    def df_dco2(self, params, Y_C02):
        """
        Returns the derivative of the model with respect to CO2.
        """
        return params[2]

    @property
    def num_params(self):
        return 3

    def __str__(self):
        return "ModelLinear"


class ModelQuadratic(Model):
    def __init__(self):
        super().__init__()

    def embedded_f(self, params, Y_C02):
        return params[1] + params[2] * Y_C02 + params[3] * Y_C02 ** 2

    def df_dco2(self, params, Y_C02):
        """
        Returns the derivative of the embedded model with respect to CO2.
        """
        return params[2] + 2 * params[3] * Y_C02

    @property
    def num_params(self):
        return 4

    def __str__(self):
        return "ModelQuadratic"


def compute_map(params):
    """Compute the MAP estimate of the parameters."""
    # Compute the MAP estimate of the parameters
    def func(x):
        alpha, f0, f1 = x
        if alpha < 0 or alpha > 1:
            return 1e10
        if f0 < 0 or f0 > 1:
            return 1e10
        l = min(-MODEL_T_LINEAR.log_likelihood(x), 1e10)
        if np.isnan(l):
            return 1e10
        return l

    res = optimize.minimize(
        func,
        params,
        bounds=optimize.Bounds([0, 0, -0.005], [1, 1, 0.005], [True, True, True]),
        method="Nelder-Mead",
        tol=1e-12
    )
    if res.success:
        print("\tOptimization was successful: ", res.x)
    else:
        print("\tOptimization failed.")
    return res


MODEL_T_CONSTANT = ModelConstant()
MODEL_T_LINEAR = ModelLinear()
MODEL_T_QUADRATIC = ModelQuadratic()
