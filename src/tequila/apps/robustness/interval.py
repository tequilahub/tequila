from abc import ABC
import numpy as np
from numbers import Real as RealNumber
from typing import List, Dict, Union, Hashable, Tuple

from tequila.apps.robustness.helpers import make_paulicliques

from tequila import simulate
from tequila.circuit import QCircuit
from tequila.circuit.noise import NoiseModel
from tequila.hamiltonian import QubitHamiltonian
from tequila.objective import Variable, ExpectationValue

__all__ = ['robustness_interval',
           'RobustnessInterval',
           'SDPInterval',
           'GramianExpectationBound',
           'GramianEigenvalueInterval']

_EXPECTATION_ALIASES = ['expectation']
_EIGENVALUE_ALIASES = ['eigenvalue', 'eigval']

_GRAMIAN_ALIASES = ['gramian', 'gram']
_METHODS = ['sdp'] + ['best'] + _GRAMIAN_ALIASES


class RobustnessInterval(ABC):
    def __init__(self,
                 fidelity: float,
                 U: QCircuit = None,
                 H: QubitHamiltonian = None,
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

        self._hamiltonian_expectation = self._precomputed_stats.get('hamiltonian_expectations')
        self._hamiltonian_variance = self._precomputed_stats.get('hamiltonian_variances')
        self._normalization_constant = self._precomputed_stats.get('normalization_const')

        self._fidelity = fidelity

        self._lower_bound = None
        self._upper_bound = None
        self._expectation = None
        self._variance = None

    def _sanity_checks(self):
        pass

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
                 precomputed_stats: Dict[str, Union[List[List[float]], List[float], float]] = None,
                 variables=None,
                 samples=None,
                 backend=None,
                 device=None,
                 noise=None,
                 group_terms=True):

        super(SDPInterval, self).__init__(fidelity, U, H, precomputed_stats)

        self._compute_stats(variables, samples, backend, device, noise, group_terms)
        self._sanity_checks()
        self._lower_bound, self._upper_bound = self._compute_interval()
        self._expectation = self._compute_expectation()

    def _sanity_checks(self):
        if self._fidelity < 0 or self._fidelity > 1:
            raise ValueError(f'encountered invalid fidelity; got {self._fidelity}, must be within [0, 1]!')

        # make sure that variance for each of the pauligroups is positive
        if self._pauligroups_variances is not None:
            for i, group_variance in enumerate(self._pauligroups_variances):
                if group_variance <= -1e-6:
                    raise ValueError(f'negative variance encountered: {group_variance}')
                self._pauligroups_variances[i] = max(0.0, group_variance)

        # make sure that variance of Hamiltonian is positive
        if self._hamiltonian_variance is not None:
            if self._hamiltonian_variance <= -1e-6:
                raise ValueError(f'negative variance encountered: v={self._hamiltonian_variance}')

            self._hamiltonian_variance = max(0.0, float(self._hamiltonian_variance))

    def _compute_stats(self, variables, samples, backend, device, noise, group_terms):
        if None not in [self._pauligroups,
                        self._pauligroups_coeffs,
                        self._pauligroups_expectations,
                        self._pauligroups_eigenvalues]:
            # here stats have been provided; no need to recompute
            return

        if self._U is None or self._H is None:
            raise ValueError('If U or H is not provided, you must provide precomputed_stats!')

        # compute expectation values
        if group_terms:
            self._pauligroups = make_paulicliques(H=self._H)
            self._pauligroups_coeffs = [ps.coeff.real for ps in self._pauligroups]
        else:
            self._pauligroups = [ps.naked() for ps in self._H.paulistrings]
            self._pauligroups_coeffs = [ps.coeff.real for ps in self._H.paulistrings]

        objectives = [ExpectationValue(U=self._U + p_str.U, H=p_str.H) for p_str in self._pauligroups]

        # compute expectation values
        self._pauligroups_expectations = [
            simulate(objective=objective, variables=variables, samples=samples, backend=backend, device=device,
                     noise=noise) for objective in objectives]

        # compute eigenvalues for each term
        self._pauligroups_eigenvalues = [p_str.compute_eigenvalues() for p_str in self._pauligroups]

    def _compute_interval(self) -> Tuple[float, float]:
        lower_bound, upper_bound = 0.0, 0.0

        for p_str, p_coeff, p_eigvals, p_expec in zip(self._pauligroups, self._pauligroups_coeffs,
                                                      self._pauligroups_eigenvalues, self._pauligroups_expectations):

            min_eigval = min(p_eigvals)
            max_eigval = max(p_eigvals)

            if str(p_str) == 'I' or len(p_str) == 0:
                pauli_lower_bound = pauli_upper_bound = 1.0
            else:
                expec_normalized = np.clip(2 * (p_expec - min_eigval) / (max_eigval - min_eigval) - 1, -1, 1,
                                           dtype=float)

                pauli_lower_bound = (max_eigval - min_eigval) / 2.0 * (
                        1 + self._calc_lower_bound(expec_normalized, self.fidelity)) + min_eigval

                pauli_upper_bound = (max_eigval - min_eigval) / 2.0 * (
                        1 + self._calc_upper_bound(expec_normalized, self.fidelity)) + min_eigval

            lower_bound += p_coeff * pauli_lower_bound if p_coeff > 0 else p_coeff * pauli_upper_bound
            upper_bound += p_coeff * pauli_upper_bound if p_coeff > 0 else p_coeff * pauli_lower_bound

        return lower_bound, upper_bound

    def _compute_expectation(self):
        return np.dot(self._pauligroups_coeffs, self._pauligroups_expectations)

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
                 precomputed_stats: Dict[str, Union[List[List[float]], List[float], float]] = None,
                 variables=None,
                 samples=None,
                 backend=None,
                 device=None,
                 noise=None,
                 group_terms=True):
        super(GramianExpectationBound, self).__init__(fidelity, U, H, precomputed_stats)

        self._compute_stats(variables, samples, backend, device, noise, group_terms)
        self._sanity_checks()
        self._lower_bound = self._compute_interval(group_terms)
        self._upper_bound = np.inf
        self._expectation = self._compute_expectation(group_terms)
        self._variance = self._compute_variance(group_terms)

    def _sanity_checks(self):
        if self._fidelity < 0 or self._fidelity > 1:
            raise ValueError(f'encountered invalid fidelity; got {self._fidelity}, must be within [0, 1]!')

        # make sure that variance for each of the pauligroups is positive
        if self._pauligroups_variances is not None:
            for i, group_variance in enumerate(self._pauligroups_variances):
                if group_variance <= -1e-6:
                    raise ValueError(f'negative variance encountered: {group_variance}')
                self._pauligroups_variances[i] = max(0.0, group_variance)

        # make sure that variance of Hamiltonian is positive
        if self._hamiltonian_variance is not None:
            if self._hamiltonian_variance <= -1e-6:
                raise ValueError(f'negative variance encountered: v={self._hamiltonian_variance}')

            self._hamiltonian_variance = max(0.0, float(self._hamiltonian_variance))

    def _compute_stats(self, variables, samples, backend, device, noise, group_terms):
        if None not in [self._pauligroups,
                        self._pauligroups_coeffs,
                        self._pauligroups_expectations,
                        self._pauligroups_variances,
                        self._pauligroups_eigenvalues]:
            # here stats have been provided; no need to recompute
            return

        if self._U is None or self._H is None:
            raise ValueError('If U or H is not provided, you must provide precomputed_stats!')

        if group_terms:
            # here we group pauli terms into groups of commuting terms
            self._pauligroups = make_paulicliques(H=self._H)
            self._pauligroups_coeffs = [ps.coeff.real for ps in self._pauligroups]
            self._pauligroups_eigenvalues = [p_str.compute_eigenvalues() for p_str in self._pauligroups]

            self._pauligroups_expectations = []
            self._pauligroups_variances = []

            # expectation for each pauli group
            objectives = [ExpectationValue(U=self._U + group.U, H=group.H) for group in self._pauligroups]
            self._pauligroups_expectations = [
                simulate(objective, variables=variables, samples=samples, backend=backend, device=device, noise=noise)
                for objective in objectives]

            # variance
            objectives = [ExpectationValue(U=self._U + group.U, H=(group.H - e) ** 2) for group, e in
                          zip(self._pauligroups, self._pauligroups_expectations)]
            self._pauligroups_variances = [
                simulate(objective, variables=variables, samples=samples, backend=backend, device=device, noise=noise)
                for objective in objectives]

        else:
            # here we compute stats for the entire hamiltonian and add a const s.t. it is ≥ 0
            pauli_strings = self._H.paulistrings
            pauli_coeffs = [ps.coeff.real for ps in pauli_strings]
            const_pauli_terms = [i for i, pstr in enumerate(pauli_strings) if len(pstr) == 0]

            self._normalization_constant = -np.sum([pauli_coeffs[i] for i in const_pauli_terms])
            self._normalization_constant += np.sum([
                np.abs(pauli_coeffs[i]) for i in set(range(len(pauli_coeffs))) - set(const_pauli_terms)])

            # compute expectation
            objective = ExpectationValue(U=self._U, H=self._H)
            self._hamiltonian_expectation = simulate(objective, variables=variables, samples=samples, backend=backend,
                                                     device=device, noise=noise)

            # compute variance
            objective = ExpectationValue(U=self._U, H=(self._H - self._hamiltonian_expectation) ** 2)
            self._hamiltonian_variance = simulate(objective, variables=variables, samples=samples, backend=backend,
                                                  device=device, noise=noise)

    def _compute_interval(self, group_terms) -> float:
        if group_terms:
            return self._compute_bound_grouped()
        else:
            assert self._normalization_constant is not None
            return self._compute_bound_hamiltonian()

    def _compute_bound_hamiltonian(self) -> float:
        bound = -self._normalization_constant + self._calc_lower_bound(
            self._normalization_constant + self._hamiltonian_expectation, self._hamiltonian_variance, self.fidelity)
        return bound

    def _compute_bound_grouped(self) -> float:
        bound = 0.0

        for eigvals, expec, variance in zip(self._pauligroups_eigenvalues, self._pauligroups_expectations,
                                            self._pauligroups_variances):
            min_eigval = min(eigvals)
            expec_pos = np.clip(expec - min_eigval, 0, None, dtype=float)
            bound += min_eigval + self._calc_lower_bound(expec_pos, variance, self.fidelity)

        return bound

    def _compute_expectation(self, group_tems) -> Union[float, RealNumber]:
        if group_tems:
            return float(np.dot(self._pauligroups_coeffs, self._pauligroups_expectations))
        else:
            return self._hamiltonian_expectation

    def _compute_variance(self, group_tems) -> Union[float, RealNumber]:
        if group_tems:
            return np.nan
        else:
            return self._hamiltonian_variance

    @staticmethod
    def _calc_lower_bound(expectation, variance, fidelity):
        assert expectation >= 0

        if fidelity / (1 - fidelity) < variance / (expectation ** 2):
            return 0.0

        return fidelity * expectation + (1 - fidelity) / expectation * variance - 2 * np.sqrt(
            variance * fidelity * (1 - fidelity))

    @staticmethod
    def _calc_upper_bound():
        return np.inf


class GramianEigenvalueInterval(RobustnessInterval):

    def __init__(self,
                 fidelity: float,
                 U: QCircuit = None,
                 H: QubitHamiltonian = None,
                 precomputed_stats: Dict[str, Union[List[List[float]], List[float], float]] = None,
                 variables=None,
                 samples=None,
                 backend=None,
                 device=None,
                 noise=None):
        super(GramianEigenvalueInterval, self).__init__(fidelity, U, H, precomputed_stats)

        self._compute_stats(variables, samples, backend, device, noise)
        self._sanity_checks()
        self._lower_bound, self._upper_bound = self._compute_interval()
        self._expectation = self._compute_expectation()
        self._variance = self._compute_variance()

    def _sanity_checks(self):
        if self._fidelity < 0 or self._fidelity > 1:
            raise ValueError(f'encountered invalid fidelity; got {self._fidelity}, must be within [0, 1]!')

        # make sure that variance of Hamiltonian is positive
        if self._hamiltonian_variance <= -1e-6:
            raise ValueError(f'negative variance encountered: v={self._hamiltonian_variance}')

        self._hamiltonian_variance = max(0.0, float(self._hamiltonian_variance))

    def _compute_stats(self, variables, samples, backend, device, noise):
        if None not in [self._hamiltonian_expectation, self._hamiltonian_variance]:
            # here stats have been provided; no need to recompute
            return

        if self._U is None or self._H is None:
            raise ValueError('If U or H is not provided, you must provide precomputed_stats!')

        # compute expectation
        objective = ExpectationValue(U=self._U, H=self._H)
        self._hamiltonian_expectation = simulate(objective, variables=variables, samples=samples, backend=backend,
                                                 device=device, noise=noise)

        # compute variance
        objective = ExpectationValue(U=self._U, H=(self._H - self._hamiltonian_expectation) ** 2)
        self._hamiltonian_variance = simulate(objective, variables=variables, samples=samples, backend=backend,
                                              device=device, noise=noise)

    def _compute_interval(self) -> Tuple[float, float]:
        lower_bound = self._calc_lower_bound(self._hamiltonian_expectation, self._hamiltonian_variance, self.fidelity)
        upper_bound = self._calc_upper_bound(self._hamiltonian_expectation, self._hamiltonian_variance, self.fidelity)
        return lower_bound, upper_bound

    def _compute_expectation(self) -> Union[float, RealNumber]:
        return self._hamiltonian_expectation

    def _compute_variance(self) -> Union[float, RealNumber]:
        return self._hamiltonian_variance

    @staticmethod
    def _calc_lower_bound(expectation, variance, fidelity) -> float:
        return expectation - np.sqrt(variance) * np.sqrt((1.0 - fidelity) / fidelity)

    @staticmethod
    def _calc_upper_bound(expectation, variance, fidelity) -> float:
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
                        return_object: bool = False) -> (float, float, float, Union[RobustnessInterval, None]):
    """ calculate robustness intervals

    Parameters
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
    return_object: bool, optional:
        if set to True, then an instance of RobustnessInterval is returned. This contains additional information used
        in the calculation of the interval, such as the (sampled, simulated) expectation value and variance of H.

    Returns
    -------
        tuple with ((float, float, float), RobustnessInterval or None) where (float, float, float) is the lower bound,
        expectation of H with U, and the upper bound.

    """
    method = method.lower().replace('_', '').replace(' ', '')
    kind = kind.lower().replace('_', '').replace(' ', '')

    if kind not in _EXPECTATION_ALIASES + _EIGENVALUE_ALIASES:
        raise ValueError(f'unknown robustness interval type; got {kind}, '
                         + f'must be one of {_EXPECTATION_ALIASES + _EIGENVALUE_ALIASES}')

    if method not in _METHODS:
        raise ValueError(f'unknown method; got {method}, must be one of {_METHODS}')

    if method == 'sdp':
        interval = SDPInterval(U=U, H=H, fidelity=fidelity, variables=variables, backend=backend, noise=noise,
                               device=device, samples=samples, group_terms=group_terms)
        return interval.interval, (interval if return_object else None)

    if method in _GRAMIAN_ALIASES:

        if kind in _EXPECTATION_ALIASES:
            interval = GramianExpectationBound(U=U, H=H, fidelity=fidelity, variables=variables, backend=backend,
                                               noise=noise, device=device, samples=samples, group_terms=group_terms)
            return interval.interval, (interval if return_object else None)

        if kind in _EIGENVALUE_ALIASES:
            interval = GramianEigenvalueInterval(U=U, H=H, fidelity=fidelity, variables=variables, backend=backend,
                                                 noise=noise, device=device, samples=samples)
            return interval.interval, (interval if return_object else None)

    if method == 'best':
        sdp_interval = SDPInterval(U=U, H=H, fidelity=fidelity, variables=variables, backend=backend, noise=noise,
                                   device=device, samples=samples, group_terms=True)
        gramian_exp_interval = GramianExpectationBound(U=U, H=H, fidelity=fidelity, variables=variables,
                                                       backend=backend, noise=noise, device=device, samples=samples,
                                                       group_terms=True)
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
            gramian_eigv_interval = GramianEigenvalueInterval(U=U, H=H, fidelity=fidelity, variables=variables,
                                                              backend=backend, noise=noise, device=device,
                                                              samples=samples)

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
