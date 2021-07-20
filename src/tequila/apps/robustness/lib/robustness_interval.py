from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats
from typing import *

EXPECTATION_INTERVAL_TYPES = ['expectation']
EIGENVALUE_INTERVAL_TYPES = ['eigenvalue', 'eigenvalues', 'eigval', 'eigvals']

ROBUSTNESS_INTERVAL_TYPES = EXPECTATION_INTERVAL_TYPES + EIGENVALUE_INTERVAL_TYPES
ROBUSTNESS_INTERVAL_METHODS = ['sdp', 'gramian', 'gram']


class RobustnessInterval(ABC):
    def __init__(self, statistics: Dict[str, Union[List[List[float]], List[float], float]], fidelity: float,
                 normalization_const: Union[float, None], confidence_level):
        # clean stats keys
        statistics = {k.replace('_', '').replace(' ', ''): v for k, v in statistics.items()}

        self._expectations = statistics.get('expectationvalues')
        self._variances = statistics.get('variances')
        self._pauli_strings = statistics.get('paulistrings')
        self._pauli_coeffs = statistics.get('paulicoeffs')
        self._pauli_eigenvalues = statistics.get('paulieigenvalues')

        self._fidelity = fidelity
        self._confidence_level = confidence_level

        self._lower_bound = None
        self._upper_bound = None

        self._mean_expectation = None
        self._mean_variance = None

        self._compute_interval(normalization_const=normalization_const)

    @abstractmethod
    def _compute_interval(self, *args, **kwargs):
        pass

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def expectations(self):
        return self._expectations

    @property
    def mean_expectation(self):
        return np.mean(self._expectations) if self._mean_expectation is None else self._mean_expectation

    @property
    def variances(self):
        return self._variances

    @property
    def mean_variance(self):
        return np.mean(self._variances) if self._mean_variance is None else self._mean_variance

    @property
    def fidelity(self):
        return self._fidelity


class SDPInterval(RobustnessInterval):

    def _compute_interval(self, *args, **kwargs):

        bounds = np.array([self._compute_interval_single(
            self._pauli_strings, self._pauli_coeffs, self._pauli_eigenvalues, pauli_expecs)
            for pauli_expecs in self.expectations
        ])

        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]

        n_reps = len(lower_bounds)

        if n_reps == 1:
            self._lower_bound = lower_bounds[0]
            self._upper_bound = upper_bounds[0]
        else:
            # one sided confidence interval for lower bound
            lower_bound_mean = np.mean(lower_bounds)
            lower_bound_variance = np.var(lower_bounds, ddof=1, dtype=np.float64)
            self._lower_bound = lower_bound_mean - np.sqrt(lower_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._confidence_level, df=n_reps - 1)

            # one sided confidence interval for upper bound
            upper_bound_mean = np.mean(upper_bounds)
            upper_bound_variance = np.var(upper_bounds, ddof=1, dtype=np.float64)
            self._upper_bound = upper_bound_mean - np.sqrt(upper_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._confidence_level, df=n_reps - 1)

    def _compute_interval_single(self, pauli_strings, pauli_coeffs, pauli_eigvals, pauli_expecs):
        lower_bound = upper_bound = 0.0

        for p_str, p_coeff, p_eigvals, p_expec in zip(pauli_strings, pauli_coeffs, pauli_eigvals, pauli_expecs):

            min_eigval = min(p_eigvals)
            max_eigval = max(p_eigvals)

            if str(p_str) == 'I' or len(p_str) == 0:
                pauli_lower_bound = pauli_upper_bound = 1.0
            else:
                expec_normalized = np.clip(2 * (p_expec - min_eigval) / (max_eigval - min_eigval) - 1, -1, 1,
                                           dtype=np.float64)

                pauli_lower_bound = (max_eigval - min_eigval) / 2.0 * (
                        1 + self._calc_lower_bound(expec_normalized, self.fidelity)) + min_eigval

                pauli_upper_bound = (max_eigval - min_eigval) / 2.0 * (
                        1 + self._calc_upper_bound(expec_normalized, self.fidelity)) + min_eigval

            lower_bound += p_coeff * pauli_lower_bound if p_coeff > 0 else p_coeff * pauli_upper_bound
            upper_bound += p_coeff * pauli_upper_bound if p_coeff > 0 else p_coeff * pauli_lower_bound

        return lower_bound, upper_bound

    @staticmethod
    def _calc_lower_bound(a, f):
        assert -1.0 <= a <= 1.0, 'data not normalized to [-1, 1]'

        if f < 0.5 * (1 - a):
            return -1.0

        return (2 * f - 1) * a - 2 * np.sqrt(f * (1 - f) * (1 - a ** 2))

    @staticmethod
    def _calc_upper_bound(a, f):
        assert -1.0 <= a <= 1.0, 'data not normalized to [-1, 1]'

        if f < 0.5 * (1 + a):
            return 1.0

        return (2.0 * f - 1.0) * a + 2.0 * np.sqrt(f * (1.0 - f) * (1.0 - a ** 2))


class GramianExpectationBound(RobustnessInterval):

    def _compute_interval(self, normalization_const, *args, **kwargs):
        lower_bounds = [normalization_const + (
                np.sqrt(self.fidelity) * (e - normalization_const) - np.sqrt(1 - self.fidelity) * np.sqrt(
            v)) ** 2 / (e - normalization_const) for e, v in zip(self.expectations, self.variances)]

        n_reps = len(lower_bounds)

        if n_reps == 1:
            self._lower_bound = lower_bounds[0]
        else:
            # one sided confidence interval for lower bound
            lower_bound_mean = np.mean(lower_bounds)
            lower_bound_variance = np.var(lower_bounds, ddof=1, dtype=np.float64)
            self._lower_bound = lower_bound_mean - np.sqrt(lower_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._confidence_level, df=n_reps - 1)


class GramianEigenvalueInterval(RobustnessInterval):

    def _compute_interval(self, *args, **kwargs):

        if self.fidelity <= 0:
            self._lower_bound = -np.inf
            self._upper_bound = np.inf
            return

        lower_bounds = [e - np.sqrt(v) * np.sqrt((1 - self.fidelity) / self.fidelity)
                        for e, v in zip(self.expectations, self.variances)]
        upper_bounds = [e + np.sqrt(v) * np.sqrt((1 - self.fidelity) / self.fidelity)
                        for e, v in zip(self.expectations, self.variances)]

        n_reps = len(lower_bounds)

        if n_reps == 1:
            self._lower_bound = lower_bounds[0]
            self._upper_bound = upper_bounds[0]
        else:
            # one sided confidence interval for lower bound
            lower_bound_mean = np.mean(lower_bounds)
            lower_bound_variance = np.var(lower_bounds, ddof=1, dtype=np.float64)
            self._lower_bound = lower_bound_mean - np.sqrt(lower_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._confidence_level, df=n_reps - 1)

            # one sided confidence interval for upper bound
            upper_bound_mean = np.mean(upper_bounds)
            upper_bound_variance = np.var(upper_bounds, ddof=1, dtype=np.float64)
            self._upper_bound = upper_bound_mean - np.sqrt(upper_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._confidence_level, df=n_reps - 1)


def compute_robustness_interval(method: str,
                                kind: str,
                                statistics: Dict[str, Union[List[List[float]], List[float], float]],
                                fidelity: float,
                                normalization_const: float = None,
                                confidence_level: float = 1e-2) -> RobustnessInterval:
    """
    convenience function for calculation of robustness intervals

    Args:
        method: str, method used to compute robustness interval
        kind: str, type of robustness interval
        statistics: dict, must contain statistics required for the robustness interval computation.
        fidelity: float, a lower bound to the fidelity with the target state
        confidence_level: float, defaults to 1e-2; only relevant when statistics were sampled
        normalization_const: float, required when kind is set to expectation and method is gramian; ensures that A ≥ 0

    Returns:
        RobustnessInterval
    """
    method = method.lower().replace('_', '').replace(' ', '')
    kind = kind.lower().replace('_', '').replace(' ', '')

    if kind not in ROBUSTNESS_INTERVAL_TYPES:
        raise ValueError(
            f'unknown robustness interval type; got {kind}, but kind must be one of {ROBUSTNESS_INTERVAL_TYPES} ')

    if method not in ROBUSTNESS_INTERVAL_METHODS:
        raise ValueError(f'unknown method; got {method}, but method must be one of {ROBUSTNESS_INTERVAL_METHODS}')

    if kind in EXPECTATION_INTERVAL_TYPES:
        if method == 'sdp':
            return SDPInterval(statistics=statistics,
                               fidelity=fidelity,
                               normalization_const=None,
                               confidence_level=confidence_level)

        if method == 'gramian':
            if normalization_const is None:
                raise ValueError(f'normalization constant required when kind={kind} and method={method}')

            return GramianExpectationBound(statistics=statistics,
                                           fidelity=fidelity,
                                           normalization_const=normalization_const,
                                           confidence_level=confidence_level)

        raise ValueError(f'unknown method {method} for robustness interval with kind={kind}')

    if kind in EIGENVALUE_INTERVAL_TYPES:
        return GramianEigenvalueInterval(statistics=statistics,
                                         fidelity=fidelity,
                                         normalization_const=None,
                                         confidence_level=confidence_level)
