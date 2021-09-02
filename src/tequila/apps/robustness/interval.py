from abc import ABC
import numpy as np
import scipy.stats as stats
from numbers import Real as RealNumber
from typing import List, Dict, Union, Hashable
import warnings

from tequila.apps.robustness.helpers import make_paulicliques

from tequila import simulate
from tequila.circuit import QCircuit
from tequila.circuit.noise import NoiseModel
from tequila.hamiltonian import QubitHamiltonian
from tequila.objective import Variable, ExpectationValue

_EXPECTATION_ALIASES = ['expectation']
_EIGENVALUE_ALIASES = ['eigenvalue', 'eigval']

_GRAMIAN_ALIASES = ['gramian', 'gram']
_METHODS = ['sdp'] + ['best'] + _GRAMIAN_ALIASES

__all__ = ['robustness_interval',
           'RobustnessInterval',
           'SDPInterval',
           'GramianExpectationBound',
           'GramianEigenvalueInterval']


class RobustnessInterval(ABC):
    def __init__(self,
                 fidelity: float,
                 U: QCircuit = None,
                 H: QubitHamiltonian = None,
                 conf_level: float = 1e-2,
                 precomputed_stats: Dict[str, Union[List[List[float]], List[float], float]] = None):
        self._U = U
        self._H = H

        # clean stats keys
        self._precomputed_stats = {k.replace(' ', '_'): v for k, v in (precomputed_stats or {}).items()}

        self._pauligroups = self._precomputed_stats.get('pauligroups', [])
        self._pauligroups_coeffs = self._precomputed_stats.get('pauligroups_coeffs')
        self._pauligroups_eigenvalues = self._precomputed_stats.get('pauligroups_eigenvalues')
        self._pauligroups_expectations = self._precomputed_stats.get('pauligroups_expectations')
        self._pauligroups_variances = self._precomputed_stats.get('pauligroups_variances')

        self._hamiltonian_expectations = self._precomputed_stats.get('hamiltonian_expectations')
        self._hamiltonian_variances = self._precomputed_stats.get('hamiltonian_variances')
        self._normalization_constant = self._precomputed_stats.get('normalization_const')

        self._fidelity = fidelity
        self._conf_level = conf_level

        self._lower_bound = None
        self._upper_bound = None
        self._expectation = None
        self._variance = None

    def _sanity_checks(self):
        if self._fidelity < 0 or self._fidelity > 1:
            raise ValueError(f'encountered invalid fidelity; got {self._fidelity}, must be within [0, 1]!')

        # make sure that vairance is positive
        if self._pauligroups_variances is not None:
            for i, group_vars in enumerate(self._pauligroups_variances):
                for j, v in enumerate(group_vars):
                    if v <= -1e-6:
                        raise ValueError(f'negative variance encountered: {v}')

                    group_vars[j] = max(0, v)

                self._pauligroups_variances[i] = group_vars

        if self._hamiltonian_variances is not None:
            for i, v in enumerate(self._hamiltonian_variances):
                if v <= -1e-6:
                    raise ValueError(f'negative variance encountered: v={v}')

                self._hamiltonian_variances[i] = max(0, v)

    def _compute_interval(self, *args, **kwargs):
        pass

    def _compute_stats(self, **kwargs):
        pass

    def _calc_lower_bound(self, **kwargs):
        pass

    def _calc_upper_bound(self, **kwargs):
        pass

    @property
    def interval(self):
        return self.lower_bound, self.expectation, self.upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def expectation(self):
        return self._expectation

    @property
    def variance(self):
        return self._variance

    @property
    def fidelity(self):
        return self._fidelity

    @property
    def U(self):
        return self._U

    @property
    def H(self):
        return self._H

    @lower_bound.setter
    def lower_bound(self, v):
        self._lower_bound = v

    @upper_bound.setter
    def upper_bound(self, v):
        self._lower_bound = v


class SDPInterval(RobustnessInterval):

    def __init__(self,
                 fidelity: float,
                 U: QCircuit = None,
                 H: QubitHamiltonian = None,
                 conf_level: float = 1e-2,
                 precomputed_stats: Dict[str, Union[List[List[float]], List[float], float]] = None,
                 variables=None,
                 samples=None,
                 backend=None,
                 device=None,
                 noise=None,
                 nreps=None,
                 group_terms=True):

        super(SDPInterval, self).__init__(fidelity, U, H, conf_level, precomputed_stats)

        self._compute_stats(variables, samples, backend, device, noise, nreps, group_terms)
        self._sanity_checks()
        self._lower_bound, self._upper_bound = self._compute_interval()
        self._expectation = self._compute_expectation()

    def _compute_stats(self, variables, samples, backend, device, noise, nreps, group_terms):
        if None not in [self._pauligroups, self._pauligroups_coeffs, self._pauligroups_expectations,
                        self._pauligroups_eigenvalues]:
            # here we assume that stats have been provided; no need to recompute
            return

        if self._U is None or self._H is None:
            raise ValueError('If U or H is not provided, you must provide precomputed_stats!')

        if samples is None:
            nreps = 1

        # compute expectation values
        if group_terms:
            self._pauligroups = make_paulicliques(H=self._H)
            self._pauligroups_coeffs = [ps.coeff.real for ps in self._pauligroups]
        else:
            self._pauligroups = [ps.naked() for ps in self._H.paulistrings]
            self._pauligroups_coeffs = [ps.coeff.real for ps in self._H.paulistrings]

        objectives = [ExpectationValue(U=self._U + p_str.U, H=p_str.H) for p_str in self._pauligroups]

        # compute expectation values
        self._pauligroups_expectations = [[simulate(objective=objective,  # noqa
                                                    variables=variables,
                                                    samples=samples,
                                                    backend=backend,
                                                    device=device,
                                                    noise=noise) for objective in objectives] for _ in range(nreps)]

        # compute eigenvalues for each term
        self._pauligroups_eigenvalues = [p_str.compute_eigenvalues() for p_str in self._pauligroups]

    def _compute_interval(self, *args, **kwargs):

        bounds = np.array([self._compute_interval_single(
            pauligroups=self._pauligroups,
            pauligroups_coeffs=self._pauligroups_coeffs,
            pauligroups_eigvals=self._pauligroups_eigenvalues,
            pauligroups_expectations=pauli_expecs
        ) for pauli_expecs in self._pauligroups_expectations])

        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]

        n_reps = len(lower_bounds)

        if n_reps == 1:
            lower_bound, upper_bound = lower_bounds[0], upper_bounds[0]
        else:
            # one sided confidence interval for lower bound
            lower_bound_mean = np.mean(lower_bounds)
            lower_bound_variance = np.var(lower_bounds, ddof=1, dtype=np.float64)
            lower_bound = lower_bound_mean - np.sqrt(lower_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._conf_level, df=n_reps - 1)

            # one sided confidence interval for upper bound
            upper_bound_mean = np.mean(upper_bounds)
            upper_bound_variance = np.var(upper_bounds, ddof=1, dtype=np.float64)
            upper_bound = upper_bound_mean - np.sqrt(upper_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._conf_level, df=n_reps - 1)

        return lower_bound, upper_bound

    def _compute_interval_single(self, pauligroups, pauligroups_coeffs, pauligroups_eigvals, pauligroups_expectations):
        lower_bound, upper_bound = 0.0, 0.0

        for p_str, p_coeff, p_eigvals, p_expec in zip(pauligroups, pauligroups_coeffs, pauligroups_eigvals,
                                                      pauligroups_expectations):

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

    def _compute_expectation(self):
        # compute expectations
        expectation_samples = []
        for pauligroups_expecs in self._pauligroups_expectations:
            expectation_samples.append(np.dot(self._pauligroups_coeffs, pauligroups_expecs))

        mean_expectation = np.mean(expectation_samples)

        return mean_expectation

    @staticmethod
    def _calc_lower_bound(e, f):
        assert -1.0 <= e <= 1.0, 'expectation not normalized to [-1, 1]'

        if f < 0.5 * (1 - e):
            return -1.0

        return (2 * f - 1) * e - 2 * np.sqrt(f * (1 - f) * (1 - e ** 2))

    @staticmethod
    def _calc_upper_bound(e, f):
        assert -1.0 <= e <= 1.0, 'expectation not normalized to [-1, 1]'

        if f < 0.5 * (1 + e):
            return 1.0

        return (2.0 * f - 1.0) * e + 2.0 * np.sqrt(f * (1.0 - f) * (1.0 - e ** 2))


class GramianExpectationBound(RobustnessInterval):

    def __init__(self,
                 fidelity: float,
                 U: QCircuit = None,
                 H: QubitHamiltonian = None,
                 conf_level: float = 1e-2,
                 precomputed_stats: Dict[str, Union[List[List[float]], List[float], float]] = None,
                 variables=None,
                 samples=None,
                 backend=None,
                 device=None,
                 noise=None,
                 nreps=None,
                 group_terms=True):
        super(GramianExpectationBound, self).__init__(fidelity, U, H, conf_level, precomputed_stats)

        self._compute_stats(variables, samples, backend, device, noise, nreps, group_terms)
        self._sanity_checks()
        self._lower_bound = self._compute_interval(group_terms)
        self._upper_bound = np.inf
        self._expectation = self._compute_expectation(group_terms)
        self._variance = self._compute_variance(group_terms)

    def _compute_stats(self, variables, samples, backend, device, noise, nreps, group_terms):
        if None not in [self._pauligroups, self._pauligroups_coeffs, self._pauligroups_expectations,
                        self._pauligroups_variances, self._pauligroups_eigenvalues]:
            # here we assume that stats have been provided; no need to recompute
            return

        if self._U is None or self._H is None:
            raise ValueError('If U or H is not provided, you must provide precomputed_stats!')

        if samples is None:
            nreps = 1

        if group_terms:
            # here we group pauli terms into groups of commuting terms
            self._pauligroups = make_paulicliques(H=self._H)
            self._pauligroups_coeffs = [ps.coeff.real for ps in self._pauligroups]
            self._pauligroups_eigenvalues = [p_str.compute_eigenvalues() for p_str in self._pauligroups]

            objectives = [ExpectationValue(U=self._U + group.U, H=group.H) for group in self._pauligroups]

            self._pauligroups_expectations = []
            self._pauligroups_variances = []

            for _ in range(nreps):
                # expectation
                group_expecs = [simulate(objective=objective,  # noqa
                                         variables=variables,
                                         samples=samples,
                                         backend=backend,
                                         device=device,
                                         noise=noise) for objective in objectives]

                # variance
                group_vars = [simulate(ExpectationValue(U=self._U + group.U, H=(group.H - e) ** 2),  # noqa
                                       variables=variables,
                                       samples=samples,
                                       backend=backend,
                                       device=device,
                                       noise=noise) for group, e in zip(self._pauligroups, group_expecs)]

                self._pauligroups_expectations.append(group_expecs)
                self._pauligroups_variances.append(group_vars)

        else:
            # here we compute stats for the entire hamiltonian and add a const than s.t. it is ≥ 0
            pauli_strings = self._H.paulistrings
            pauli_coeffs = [ps.coeff.real for ps in pauli_strings]
            const_pauli_terms = [i for i, pstr in enumerate(pauli_strings) if len(pstr) == 0]

            self._normalization_constant = -np.sum([pauli_coeffs[i] for i in const_pauli_terms])
            self._normalization_constant += np.sum([
                np.abs(pauli_coeffs[i]) for i in set(range(len(pauli_coeffs))) - set(const_pauli_terms)])

            objective = ExpectationValue(U=self._U, H=self._H)

            # compute expectation reps
            self._hamiltonian_expectations = [simulate(objective,  # noqa
                                                       variables=variables,
                                                       samples=samples,
                                                       backend=backend,
                                                       device=device,
                                                       noise=noise) for _ in range(nreps)]

            # compute variance reps
            self._hamiltonian_variances = [simulate(ExpectationValue(U=self._U, H=(self._H - e) ** 2),  # noqa
                                                    variables=variables,
                                                    samples=samples,
                                                    backend=backend,
                                                    device=device,
                                                    noise=noise) for e in self._hamiltonian_expectations]

    def _compute_interval(self, group_terms):
        if group_terms:
            bounds = self._compute_bounds_grouped()
        else:
            assert self._normalization_constant is not None
            bounds = self._compute_bounds_hamiltonian()

        n_reps = len(bounds)

        if n_reps == 1:
            bound = bounds[0]
        else:
            # one sided confidence interval for lower bound
            bound_mean = np.mean(bounds)
            bound_variance = np.var(bounds, ddof=1, dtype=np.float64)
            bound = bound_mean - np.sqrt(bound_variance / n_reps) * stats.t.ppf(q=1 - self._conf_level, df=n_reps - 1)

        return bound

    def _compute_bounds_hamiltonian(self):
        bounds = []

        for expectation, variance in zip(self._hamiltonian_expectations, self._hamiltonian_variances):
            bound = -self._normalization_constant + self._calc_lower_bound(
                e=self._normalization_constant + expectation, v=variance, f=self.fidelity)  # noqa
            bounds.append(bound)

        return bounds

    def _compute_bounds_grouped(self):
        bounds = []

        for pg_expectations, pg_variances in zip(self._pauligroups_expectations, self._pauligroups_variances):

            bound = 0.0
            for eigvals, expec, variance in zip(self._pauligroups_eigenvalues, pg_expectations, pg_variances):
                min_eigval = min(eigvals)
                expec_pos = np.clip(expec - min_eigval, 0, None, dtype=np.float)
                group_lower_bound = min_eigval + self._calc_lower_bound(expec_pos, variance, self.fidelity)

                bound += group_lower_bound

            bounds.append(bound)

        return bounds

    def _compute_expectation(self, group_tems):
        if group_tems:
            expectation_samples = []
            for pauligroups_expecs in self._pauligroups_expectations:
                expectation_samples.append(np.dot(self._pauligroups_coeffs, pauligroups_expecs))
        else:
            expectation_samples = self._hamiltonian_expectations

        mean_expectation = np.mean(expectation_samples)
        return mean_expectation

    def _compute_variance(self, group_tems):
        if group_tems:
            return np.nan
        else:
            mean_variance = np.mean(self._hamiltonian_variances)
            return mean_variance

    @staticmethod
    def _calc_lower_bound(e, v, f):
        assert e >= 0

        if f / (1 - f) < v / (e ** 2):
            return 0.0

        return f * e + (1 - f) / e * v - 2 * np.sqrt(v * f * (1 - f))

    @staticmethod
    def _calc_upper_bound():
        return np.inf


class GramianEigenvalueInterval(RobustnessInterval):

    def __init__(self,
                 fidelity: float,
                 U: QCircuit = None,
                 H: QubitHamiltonian = None,
                 conf_level: float = 1e-2,
                 precomputed_stats: Dict[str, Union[List[List[float]], List[float], float]] = None,
                 variables=None,
                 samples=None,
                 backend=None,
                 device=None,
                 noise=None,
                 nreps=None):
        super(GramianEigenvalueInterval, self).__init__(fidelity, U, H, conf_level, precomputed_stats)

        self._compute_stats(variables, samples, backend, device, noise, nreps)
        self._sanity_checks()
        self._lower_bound, self._upper_bound = self._compute_interval()
        self._expectation = self._compute_expectation()
        self._variance = self._compute_variance()

    def _compute_stats(self, variables, samples, backend, device, noise, nreps):
        if None not in [self._hamiltonian_expectations, self._hamiltonian_variances]:
            # here we assume that stats have been provided; no need to recompute
            return

        if self._U is None or self._H is None:
            raise ValueError('If U or H is not provided, you must provide precomputed_stats!')

        if samples is None:
            nreps = 1

        objective = ExpectationValue(U=self._U, H=self._H)

        # compute expectation for nreps repetitions
        expectation_values = [simulate(objective,  # noqa
                                       variables=variables,
                                       samples=samples,
                                       backend=backend,
                                       device=device,
                                       noise=noise) for _ in range(nreps)]

        # compute variance for nreps repetitions
        variances = [simulate(ExpectationValue(U=self._U, H=(self._H - e) ** 2),  # noqa
                              variables=variables,
                              samples=samples,
                              backend=backend,
                              device=device,
                              noise=noise) for e in expectation_values]

        self._hamiltonian_expectations = expectation_values
        self._hamiltonian_variances = variances

    def _compute_interval(self):

        lower_bounds = [self._calc_lower_bound(e, v, self.fidelity)
                        for e, v in zip(self._hamiltonian_expectations, self._hamiltonian_variances)]
        upper_bounds = [self._calc_upper_bound(e, v, self.fidelity)
                        for e, v in zip(self._hamiltonian_expectations, self._hamiltonian_variances)]

        n_reps = len(lower_bounds)

        if n_reps == 1:
            lower_bound = lower_bounds[0]
            upper_bound = upper_bounds[0]
        else:
            # one sided confidence interval for lower bound
            lower_bound_mean = np.mean(lower_bounds)
            lower_bound_variance = np.var(lower_bounds, ddof=1, dtype=np.float64)
            lower_bound = lower_bound_mean - np.sqrt(lower_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._conf_level, df=n_reps - 1)

            # one sided confidence interval for upper bound
            upper_bound_mean = np.mean(upper_bounds)
            upper_bound_variance = np.var(upper_bounds, ddof=1, dtype=np.float64)
            upper_bound = upper_bound_mean - np.sqrt(upper_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._conf_level, df=n_reps - 1)

        return lower_bound, upper_bound

    def _compute_expectation(self):
        return np.mean(self._hamiltonian_expectations)

    def _compute_variance(self):
        return np.mean(self._hamiltonian_variances)

    @staticmethod
    def _calc_lower_bound(e, v, f):
        return e - np.sqrt(v) * np.sqrt((1.0 - f) / f)

    @staticmethod
    def _calc_upper_bound(expectation, variance, fidelity):
        if variance < 0:
            warnings.warn('CAUTION! negative variance encountered; setting to 0.0; this can lead to unreliable results')
            print('CAUTION! negative variance encountered; setting to 0.0; this can lead to unreliable results')
            variance = 0.0

        return expectation + np.sqrt(variance) * np.sqrt((1.0 - fidelity) / fidelity)


def robustness_interval(U: QCircuit,
                        H: QubitHamiltonian,
                        fidelity: float,
                        variables: Dict[Union[Variable, Hashable], RealNumber] = None,
                        kind: str = 'expectation',
                        method: str = 'best',
                        group_terms: bool = True,
                        samples: int = None,
                        backend: str = None,
                        noise: Union[str, NoiseModel] = None,
                        device: str = None,
                        nreps: int = 1,
                        conf_level: float = 1e-2,
                        return_object: bool = False) -> (float, float, float, Union[RobustnessInterval, None]):
    """ calculate robustness intervals

    Paramters
    ---------
    U: QCircuit:
        A circuit for preparing a state
    H: QubitHamiltonian
        a Hamiltonian for whose expectation / eigenvalue a robustness interval is computed
    variables: Dict:
        The variables of the ansatz circuit U given as dictionary
        with keys as tequila Variables/hashable types and values the corresponding real numbers
    fidelity: float:
        the fidelity that the state prepared by U has with the target state
    kind: str, optional:
        kind of robustness interval, must be either for `expectation` or `eigenvalue`. Defaults to `expectation`.
    method: str, optional:
        method used to compute the interval, must be one of `sdp`, `gramian` or `best`. If `best`, then intervals for
        methods are calculated and the tightest is returned
    group_terms: bool, optional:
        If True, pauli terms are grouped into groups of commuting terms and for each group a robustness interval is
        calculated which is then aggreated to compute the final interval. Applies if kind is set to 'expectation'.
    samples: int, optional:
        number of samples
    backend: str, optional:
        backend on which the circuit is simulated
    noise: str or NoiseModel, optional:
        noise to apply to the circuit
    device: str, optional:
        the device on which the circuit should (perhaps emulatedly) sample.
    nreps: int, optional:
        number of repetitions for the interval calculation. Only applies if sampling is enabled. The interval will be
        calculated nreps times and a confidence interval for the robustness bounds will be returned.
    conf_level: float, optional:
        confidence level at which a confidence confidence interval is computed. Only applies if `nreps` > 1.
    return_object: bool, optional:
        if set to True, then an instance of RobustnessInterval is returned. This contains additional information used
        in the calculation of the interval, such as the (sampled, simulated) expectation value and variance of H.

    Returns
    -------
        tuple with ((float, float, float), RobustnessInterval or None) where (float, float, float) is the lower bound,
        expectation of H with U, and the upper bound.

    Raises
    ------

    """
    method = method.lower().replace('_', '').replace(' ', '')
    kind = kind.lower().replace('_', '').replace(' ', '')

    if kind not in _EXPECTATION_ALIASES + _EIGENVALUE_ALIASES:
        raise ValueError(f'unknown robustness interval type; got {kind}, '
                         + f'must be one of {_EXPECTATION_ALIASES + _EIGENVALUE_ALIASES}')

    if method not in _METHODS:
        raise ValueError(f'unknown method; got {method}, must be one of {_METHODS}')

    if samples is None:
        nreps = 1

    if method == 'sdp':
        interval = SDPInterval(U=U, H=H, fidelity=fidelity, conf_level=conf_level,
                               nreps=nreps, variables=variables, backend=backend, noise=noise, device=device,
                               samples=samples, group_terms=group_terms)

        return interval.interval, (interval if return_object else None)

    if method in _GRAMIAN_ALIASES:

        if kind in _EXPECTATION_ALIASES:
            interval = GramianExpectationBound(U=U, H=H, fidelity=fidelity, conf_level=conf_level,
                                               nreps=nreps, variables=variables, backend=backend, noise=noise,
                                               device=device, samples=samples, group_terms=group_terms)

            return interval.interval, (interval if return_object else None)

        if kind in _EIGENVALUE_ALIASES:
            interval = GramianEigenvalueInterval(U=U, H=H, fidelity=fidelity,
                                                 conf_level=conf_level, nreps=nreps, variables=variables,
                                                 backend=backend, noise=noise, device=device, samples=samples)

            return interval.interval, (interval if return_object else None)

    if method == 'best':
        sdp_interval = SDPInterval(U=U, H=H, fidelity=fidelity, conf_level=conf_level,
                                   nreps=nreps, variables=variables, backend=backend, noise=noise, device=device,
                                   samples=samples, group_terms=True)

        gramian_exp_interval = GramianExpectationBound(U=U, H=H, fidelity=fidelity, conf_level=conf_level, nreps=nreps,
                                                       variables=variables, backend=backend, noise=noise, device=device,
                                                       samples=samples, group_terms=True)
        if kind in _EXPECTATION_ALIASES:
            max_lower_bound = max([sdp_interval.lower_bound, gramian_exp_interval.lower_bound])
            min_upper_bound = sdp_interval.upper_bound

            interval = None
            if return_object:
                interval = RobustnessInterval(fidelity=fidelity)

                interval.__dict__.update(sdp_interval.__dict__)  # so that pauli decomposition and strings are included
                interval.__dict__.update(gramian_exp_interval.__dict__)  # variance is included; overrides expectations

                interval.lower_bound = max_lower_bound
                interval.upper_bound = min_upper_bound

            return (max_lower_bound, gramian_exp_interval.expectation, min_upper_bound), interval

        if kind in _EIGENVALUE_ALIASES:
            # reuse expectation and variance
            gramian_eigv_interval = GramianEigenvalueInterval(U=U, H=H, fidelity=fidelity, conf_level=conf_level,
                                                              nreps=nreps, variables=variables, backend=backend,
                                                              noise=noise, device=device, samples=samples)

            max_lower_bound = max([sdp_interval.lower_bound,
                                   gramian_exp_interval.lower_bound,
                                   gramian_eigv_interval.lower_bound])

            min_upper_bound = min([sdp_interval.upper_bound,
                                   gramian_eigv_interval.upper_bound])

            interval = None
            if return_object:
                interval = RobustnessInterval(fidelity=fidelity)

                interval.__dict__.update(sdp_interval.__dict__)  # so that pauli decomposition and strings are included
                interval.__dict__.update(gramian_exp_interval.__dict__)  # variance is included; overrides expectations

                interval.lower_bound = max_lower_bound
                interval.upper_bound = min_upper_bound

            return (max_lower_bound, gramian_exp_interval.expectation, min_upper_bound), interval
