from abc import ABC, abstractmethod
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


class RobustnessInterval(ABC):
    def __init__(self,
                 fidelity: float,
                 U: QCircuit = None,
                 H: QubitHamiltonian = None,
                 conf_level: float = 1e-2,
                 precomputed_stats: Dict[str, Union[List[List[float]], List[float], float]] = None, **kwargs):
        if precomputed_stats is None:
            precomputed_stats = {}

        # clean stats keys
        precomputed_stats = {k.replace('_', '').replace(' ', ''): v for k, v in precomputed_stats.items()}

        self._U = U
        self._H = H

        self._expectations_reps = precomputed_stats.get('expectationvalues')
        self._variances_reps = precomputed_stats.get('variances')
        self._pauli_strings = precomputed_stats.get('paulistrings')
        self._pauli_coeffs = precomputed_stats.get('paulicoeffs')
        self._pauli_eigenvalues = precomputed_stats.get('paulieigenvalues')
        self._pauli_expectations_repeated = precomputed_stats.get('pauliexpectations')

        self._fidelity = fidelity
        self._conf_level = conf_level

        self._lower_bound = None
        self._upper_bound = None
        self._expectation = None
        self._variance = None

        self._compute_stats(**kwargs)
        self._sanity_checks()
        self._compute_interval()

    def _sanity_checks(self):
        if self._fidelity < 0 or self._fidelity > 1:
            raise ValueError(f'encountered invalid fidelity; got {self._fidelity}, must be within [0, 1]!')

        if self._variances_reps is not None:
            for i, v in enumerate(self._variances_reps):
                if v < 0:
                    warnings.warn(message=f'CAUTION! negative variance encountered; got {v}, setting to 0.0;'
                                          + 'this can lead to unreliable results', category=RuntimeWarning)
                    v = 0.0

                self._variances_reps[i] = v

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
        if self._expectation is None:
            self._expectation = np.mean(self._expectations_reps)

        return self._expectation

    @property
    def variance(self):
        if self._variance is None:
            self._variance = np.mean(self._variances_reps)

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

    @property
    def expectations_reps(self):
        return self._expectations_reps

    @property
    def variances_reps(self):
        return self._variances_reps

    @lower_bound.setter
    def lower_bound(self, v):
        self._lower_bound = v

    @upper_bound.setter
    def upper_bound(self, v):
        self._lower_bound = v


class SDPInterval(RobustnessInterval):

    def _compute_stats(self, variables=None, samples=None, backend=None, device=None, noise=None, nreps=None,
                       group_terms=True, *args, **kwargs):
        if self._U is None or self._H is None:
            # in this case, stats must be precomputed
            if None in [self._pauli_expectations, self._pauli_strings, self._pauli_coeffs, self._pauli_eigenvalues]:
                raise ValueError('If U or H is not provided, you must provide precomputed stats!')

            return

        if samples is None:
            nreps = 1

        # compute expectation values
        if group_terms:
            paulicliques = make_paulicliques(H=self._H)
            pauli_strings = [ps.naked() for ps in paulicliques]
            pauli_coeffs = [ps.coeff.real for ps in paulicliques]
        else:
            pauli_strings = [ps.naked() for ps in self._H.paulistrings]
            pauli_coeffs = [ps.coeff.real for ps in self._H.paulistrings]

        objectives = [ExpectationValue(U=self._U + p_str.U, H=p_str.H) for p_str in pauli_strings]

        # compute expectation values
        pauli_expectations = [[simulate(
            objective=objective, variables=variables, samples=samples, backend=backend, device=device, noise=noise)
            for objective in objectives] for _ in range(nreps)]

        # compute eigenvalues for each term
        pauli_eigenvalues = [p_str.compute_eigenvalues() for p_str in pauli_strings]

        self._pauli_expectations = pauli_expectations
        self._pauli_eigenvalues = pauli_eigenvalues
        self._pauli_strings = pauli_strings
        self._pauli_coeffs = pauli_coeffs

    def _compute_interval(self, *args, **kwargs):

        bounds = np.array([self._compute_interval_single(
            self._pauli_strings, self._pauli_coeffs, self._pauli_eigenvalues, pauli_expecs)
            for pauli_expecs in self._pauli_expectations
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
                q=1 - self._conf_level, df=n_reps - 1)

            # one sided confidence interval for upper bound
            upper_bound_mean = np.mean(upper_bounds)
            upper_bound_variance = np.var(upper_bounds, ddof=1, dtype=np.float64)
            self._upper_bound = upper_bound_mean - np.sqrt(upper_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._conf_level, df=n_reps - 1)

        # compute expectations
        self._expectations_reps = [np.dot(self._pauli_coeffs, pauli_expecs) for pauli_expecs in
                                   self._pauli_expectations]

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

    def _compute_stats(self, variables=None, samples=None, backend=None, device=None, noise=None, nreps=None, *args,
                       **kwargs):
        if self._U is None or self._H is None:
            # in this case, stats must be precomputed
            if None in [self._expectations_reps, self._variances_reps]:
                raise ValueError('If U or H is not provided, you must provide precomputed stats!')

            return

        # divide into constant and non constant terms
        const_coeffs = [ps.coeff.real for ps in self._H.paulistrings if len(ps.naked()) == 0]
        other_coeffs = [ps.coeff.real for ps in self._H.paulistrings if len(ps.naked()) > 0]

        self._normalization_const = np.sum(const_coeffs) - np.sum(other_coeffs)
        objective = ExpectationValue(U=self._U, H=self._H)
        expectation_values = [simulate(objective, variables=variables, samples=samples, backend=backend,
                                       device=device, noise=noise) for _ in range(nreps)]

        # compute variance
        variances = [simulate(ExpectationValue(U=self._U, H=(self._H - e) ** 2), variables=variables, samples=samples,
                              backend=backend, device=device, noise=noise) for e in expectation_values]

        self._expectations_reps = expectation_values
        self._variances_reps = variances

    def _compute_interval(self, *args, **kwargs):
        lower_bounds = [self._calc_lower_bound(e, v, self._normalization_const, self.fidelity) for e, v in
                        zip(self._expectations_reps, self._variances_reps)]

        n_reps = len(lower_bounds)

        if n_reps == 1:
            self._lower_bound = lower_bounds[0]
        else:
            # one sided confidence interval for lower bound
            lower_bound_mean = np.mean(lower_bounds)
            lower_bound_variance = np.var(lower_bounds, ddof=1, dtype=np.float64)
            self._lower_bound = lower_bound_mean - np.sqrt(lower_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._conf_level, df=n_reps - 1)

    @staticmethod
    def _calc_lower_bound(expectation, variance, normalization_const, fidelity):
        variance = max(variance, 0.0)

        if fidelity / (1 - fidelity) < variance / ((expectation - variance) ** 2):
            return normalization_const

        return normalization_const + (np.sqrt(fidelity) * (
                expectation - normalization_const) - np.sqrt(1 - fidelity) * np.sqrt(variance)) ** 2 / (
                       expectation - normalization_const)

    @staticmethod
    def _calc_upper_bound(self, **kwargs):
        return np.inf


class GramianEigenvalueInterval(RobustnessInterval):

    def _compute_stats(self, variables=None, samples=None, backend=None, device=None, noise=None, nreps=None, *args,
                       **kwargs):
        if self._U is None or self._H is None:
            # in this case, stats must be precomputed
            if None in [self._expectations_reps, self._variances_reps]:
                raise ValueError('If U or H is not provided, you must provide precomputed stats!')
            return

        objective = ExpectationValue(U=self._U, H=self._H)
        expectation_values = [simulate(objective, variables=variables, samples=samples, backend=backend,
                                       device=device, noise=noise) for _ in range(nreps)]

        # compute variance
        variances = [simulate(ExpectationValue(U=self._U, H=(self._H - e) ** 2), variables=variables, samples=samples,
                              backend=backend, device=device, noise=noise) for e in expectation_values]

        self._expectations_reps = expectation_values
        self._variances_reps = variances

    def _compute_interval(self, *args, **kwargs):

        if self.fidelity <= 0:
            self._lower_bound = -np.inf
            self._upper_bound = np.inf
            return

        lower_bounds = [self._calc_lower_bound(e, v, self.fidelity) for e, v in
                        zip(self._expectations_reps, self._variances_reps)]
        upper_bounds = [self._calc_upper_bound(e, v, self.fidelity) for e, v in
                        zip(self._expectations_reps, self._variances_reps)]

        n_reps = len(lower_bounds)

        if n_reps == 1:
            self._lower_bound = lower_bounds[0]
            self._upper_bound = upper_bounds[0]
        else:
            # one sided confidence interval for lower bound
            lower_bound_mean = np.mean(lower_bounds)
            lower_bound_variance = np.var(lower_bounds, ddof=1, dtype=np.float64)
            self._lower_bound = lower_bound_mean - np.sqrt(lower_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._conf_level, df=n_reps - 1)

            # one sided confidence interval for upper bound
            upper_bound_mean = np.mean(upper_bounds)
            upper_bound_variance = np.var(upper_bounds, ddof=1, dtype=np.float64)
            self._upper_bound = upper_bound_mean - np.sqrt(upper_bound_variance / n_reps) * stats.t.ppf(
                q=1 - self._conf_level, df=n_reps - 1)

    @staticmethod
    def _calc_lower_bound(expectation, variance, fidelity):
        variance = max(variance, 0.0)

        return expectation - np.sqrt(variance) * np.sqrt((1.0 - fidelity) / fidelity)

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
        only applies if `method` is set to sdp. If True, pauli terms are grouped into groups of commuting terms and for
        each group a robustness interval is calculated which is then aggreated to compute the final interval
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
            interval = GramianExpectationBound(U=U, H=H, fidelity=fidelity,
                                               conf_level=conf_level, nreps=nreps, variables=variables, backend=backend,
                                               noise=noise, device=device, samples=samples)

            return interval.interval, (interval if return_object else None)

        if kind in _EIGENVALUE_ALIASES:
            interval = GramianEigenvalueInterval(U=U, H=H, fidelity=fidelity,
                                                 conf_level=conf_level, nreps=nreps, variables=variables,
                                                 backend=backend, noise=noise, device=device, samples=samples)

            return interval.interval, (interval if return_object else None)

    if method == 'best':
        sdp_interval = SDPInterval(U=U, H=H, fidelity=fidelity, conf_level=conf_level,
                                   nreps=nreps, variables=variables, backend=backend, noise=noise, device=device,
                                   samples=samples)

        gramian_exp_interval = GramianExpectationBound(U=U, H=H, fidelity=fidelity,
                                                       conf_level=conf_level, nreps=nreps, variables=variables,
                                                       backend=backend, noise=noise, device=device, samples=samples)
        if kind in _EXPECTATION_ALIASES:
            max_lower_bound = max([sdp_interval.lower_bound, gramian_exp_interval.lower_bound])
            min_upper_bound = sdp_interval.upper_bound

            obj = None
            if return_object:
                obj = RobustnessInterval(fidelity=fidelity)

                obj.__dict__.update(sdp_interval.__dict__)  # so that pauli decomposition and strings are included
                obj.__dict__.update(gramian_exp_interval.__dict__)  # variance is included; overrides expectations

                obj.lower_bound = max_lower_bound
                obj.upper_bound = min_upper_bound

            return (max_lower_bound, gramian_exp_interval.expectation, min_upper_bound), obj

        if kind in _EIGENVALUE_ALIASES:
            # reuse expectation and variance
            precomputed_stats = {'expectationvalues': gramian_exp_interval.expectations_reps,
                                 'variances': gramian_exp_interval.variances_reps}

            gramian_eigv_interval = GramianEigenvalueInterval(fidelity=fidelity,
                                                              precomputed_stats=precomputed_stats,
                                                              conf_level=conf_level)

            max_lower_bound = max(
                [sdp_interval.lower_bound, gramian_exp_interval.lower_bound, gramian_eigv_interval.lower_bound])
            min_upper_bound = min(
                [sdp_interval.upper_bound, gramian_eigv_interval.upper_bound])

            obj = None
            if return_object:
                obj = RobustnessInterval(fidelity=fidelity)

                obj.__dict__.update(sdp_interval.__dict__)  # so that pauli decomposition and strings are included
                obj.__dict__.update(gramian_exp_interval.__dict__)  # variance is included; overrides expectations

                obj.lower_bound = max_lower_bound
                obj.upper_bound = min_upper_bound

            return (max_lower_bound, gramian_exp_interval.expectation, min_upper_bound), obj


if __name__ == '__main__':
    import tequila as tq
    from tequila.circuit.noise import DepolarizingError

    geometry = 'H .0 .0 .0\nH .0 .0 0.75'
    mol = tq.Molecule(geometry=geometry, basis_set='sto-3g', transformation='JORDANWIGNER')

    backend_options = dict(backend='qiskit', samples=8192, noise=DepolarizingError(0.001, level=2))
    # backend_options = dict(backend='qulacs')

    H = mol.make_hamiltonian()
    U = mol.make_upccgsd_ansatz()
    E = tq.ExpectationValue(U=U, H=H)

    result = tq.minimize(E, maxiter=0, **backend_options)

    # compute fidelity with ground state
    eigvals, eigvecs = np.linalg.eigh(H.to_matrix())
    ground_state_vec = eigvecs[:, 0]

    if backend_options.get('samples') is None:
        ground_state_wfn = tq.QubitWaveFunction.from_array(ground_state_vec)
        U_wfn = tq.simulate(U, variables=result.variables, **backend_options)
        fidelity = abs(ground_state_wfn.inner(U_wfn)) ** 2
    else:
        exact_wfn = tq.QubitWaveFunction.from_array(ground_state_vec)
        exact_wfn = tq.paulis.Projector(wfn=exact_wfn)
        fidelity = tq.ExpectationValue(U=U, H=exact_wfn)
        fidelity = tq.simulate(objective=fidelity, variables=result.variables, **backend_options)

    intervals, interval_obj = robustness_interval(U=U, H=H, fidelity=fidelity, kind='expectation', method='best',
                                                  variables=result.variables, return_object=True, **backend_options)

    print(eigvals[0], intervals, interval_obj.expectation, interval_obj.variance)
