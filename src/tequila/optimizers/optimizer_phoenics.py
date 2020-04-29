from tequila.objective.objective import Objective
from tequila.optimizers.optimizer_base import Optimizer
import typing
import numbers
from tequila.objective.objective import Variable
import copy
import warnings
import pickle
import time
from tequila import TequilaException

warnings.simplefilter("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore")
import phoenics

import numpy as np
from numpy import pi as pi
from tequila.simulators.simulator_api import compile_objective
import os
from collections import namedtuple

warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=FutureWarning)
PhoenicsReturnType = namedtuple('PhoenicsReturnType', 'energy angles history observations')

import sys


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


### wrapper for Phoenics, so that it can be used as an optimizer for parameters.
class OptimizerPhoenics(Optimizer):

    @classmethod
    def available_methods(cls):
        return "phoenics"

    def __init__(self, maxiter, backend=None, save_history=True, minimize=True, samples=None, silent=None):
        self._minimize = minimize
        if samples is not None:
            print('warning you: the samples you input do not matter, except when calling')
        super().__init__(backend=backend, maxiter=maxiter, samples=None, save_history=save_history, silent=silent)

    def _process_for_sim(self, recommendation, passives):
        '''
        renders a set of recommendations usable by the QCircuit as a list of parameter sets to choose from.
        '''
        rec = copy.deepcopy(recommendation)
        for part in rec:
            for k, v in part.items():
                part[k] = v.item()
            if passives is not None:
                for k, v in passives.items():
                    part[k] = v
        return rec

    def _process_for_phoenics(self, pset, result, passives=None):
        new = copy.deepcopy(pset)
        for k, v in new.items():
            new[k] = np.array([v], dtype=np.float32)
        if passives is not None:
            for k in passives.keys():
                del new[k]
        new['Energy'] = result

        return new

    def _make_phoenics_object(self, objective, passives=None, conf=None, *args, **kwargs):
        if conf is not None:
            if hasattr(conf, 'readlines'):
                bird = phoenics.Phoenics(config_file=conf)
            else:
                bird = phoenics.Phoenics(config_dict=conf)

            return bird
        op = objective.extract_variables()
        if passives is not None:
            for i, thing in enumerate(op):
                if thing in passives.keys():
                    op.remove(thing)

        config = {"general": {"auto_desc_gen": "False", "batches": 5, "boosted": "False", "parallel": "False"}}
        config['parameters'] = [
            {'name': k, 'periodic': 'True', 'type': 'continuous', 'size': 1, 'low': 0, 'high': 2 * pi} for k in op]
        if self._minimize is True:
            config['objectives'] = [{"name": "Energy", "goal": "minimize"}]
        else:
            config['objectives'] = [{"name": "Energy", "goal": "maximize"}]

        for k,v in kwargs.items():
            if hasattr(k, "lower") and k.lower() in config["general"]:
                config["general"][k.lower()] = v

        if not self.silent:
            print("Phoenics config:\n")
            print(config)
        bird = phoenics.Phoenics(config_dict=config)
        return bird

    def __call__(self, objective: Objective,
                 maxiter: int = None,
                 passives: typing.Dict[Variable, numbers.Real] = None,
                 samples: int = None,
                 backend: str = None,
                 noise=None,
                 previous=None,
                 phoenics_config=None,
                 save_to_file=False,
                 file_name=None,
                 *args,
                 **kwargs):

        backend_options = {}
        if 'backend_options' in kwargs:
            backend_options = kwargs['backend_options']

        if maxiter is None:
            maxiter = 10

        bird = self._make_phoenics_object(objective, passives, phoenics_config, *args, **kwargs)
        if previous is not None:
            if type(previous) is str:
                try:
                    obs = pickle.load(open(previous, 'rb'))
                except:
                    print(
                        'failed to load previous observations, which are meant to be a pickle file. Please try again or seek assistance. Starting fresh.')
                    obs = []
            elif type(previous) is list:
                if all([type(k) == dict for k in previous]):
                    obs = previous
                else:
                    print(
                        'previous observations were not in the correct format (list of dicts). Are you sure you gave me the right info? Starting fresh.')
                    obs = []

        else:
            obs = []

        if save_to_file is True:
            if type(file_name) is str:
                pass
            elif file_name is None:
                raise TequilaException(
                    'You have asked me to save phoenics observations without telling me where to do so! please provide a file_name')
            else:
                raise TequilaException('file_name must be a string!')

        ### this line below just gets the damn compiler to run, since that argument is necessary
        init = {key: np.pi for key in objective.extract_variables()}

        best = None
        best_angles = None

        # avoid multiple compilations
        compiled_objective = compile_objective(objective=objective, backend=backend, samples=samples, noise_model=noise)

        if not self.silent:
            print('phoenics has recieved')
            print("objective: \n")
            print(objective)
            print("noise model : {}".format(noise))
            print("samples     : {}".format(samples))
            print("maxiter     : {}".format(maxiter))
            print("variables   : {}".format(objective.extract_variables()))
            print("passive var : {}".format(passives))
            print("backend options {} ".format(backend), backend_options)
            print('now lets begin')
        for i in range(0, maxiter):
            with warnings.catch_warnings():
                np.testing.suppress_warnings()
                warnings.simplefilter("ignore")
                warnings.filterwarnings("ignore", category=FutureWarning)

            if len(obs) >= 1:
                precs = bird.recommend(observations=obs)
            else:
                precs = bird.recommend()

            runs = []
            recs = self._process_for_sim(precs, passives=passives)

            start = time.time()
            for i, rec in enumerate(recs):
                En = compiled_objective(variables=rec, samples=samples, noise_model=noise, **backend_options)
                runs.append((rec, En))
                if not self.silent:
                    print("energy = {:+2.8f} , angles=".format(En), rec)
            stop = time.time()
            if not self.silent:
                print("Quantum Objective evaluations: {}s Wall-Time".format(stop-start))

            for run in runs:
                angles = run[0]
                E = run[1]
                if best is None:
                    best = E
                    best_angles = angles
                else:
                    if self._minimize:
                        if E < best:
                            best = E
                            best_angles = angles
                    else:
                        if E > best:
                            best = E
                            best_angles = angles

                if self.save_history:
                    self.history.energies.append(E)
                    self.history.angles.append(angles)
                obs.append(self._process_for_phoenics(angles, E, passives=passives))

        if save_to_file is True:
            with open(file_name, 'wb') as file:
                pickle.dump(obs, file)

        if not self.silent:
            print("best energy after {} iterations : {:+2.8f}".format(self.maxiter, best))
        return PhoenicsReturnType(energy=best, angles=best_angles, history=self.history, observations=obs)


def minimize(objective: Objective,
             maxiter: int = None,
             samples: int = None,
             variables: typing.List = None,
             initial_values: typing.Dict = None,
             backend: str = None,
             noise=None,
             previous: typing.Union[str, list] = None,
             phoenics_config: typing.Union[str, typing.Dict] = None,
             save_to_file: bool = False,
             file_name: str = None,
             silent: bool = False,
             *args,
             **kwargs):
    """

    Parameters
    ----------
    objective: Objective :
        The tequila objective to optimize
    initial_values: typing.Dict[typing.Hashable, numbers.Real]: (Default value = None):
        Initial values as dictionary of Hashable types (variable keys) and floating point numbers. If given None they will all be set to zero
    variables: typing.List[typing.Hashable] :
         (Default value = None)
         List of Variables to optimize
    samples: int :
         (Default value = None)
         samples/shots to take in every run of the quantum circuits (None activates full wavefunction simulation)
    maxiter: int :
         how many iterations of phoenics to run. Note that this is NOT identical to the number of times the circuit will run.
    backend: str :
         (Default value = None)
         Simulator backend, will be automatically chosen if set to None
    noise: NoiseModel :
         (Default value = None)
         a noise model to apply to the circuits of Objective.
    previous:
        (Default value = None)
        Previous phoenics observations. If string, the name of a file from which to load them. Else, a list.
    phoenics_config:
        (Default value = None)
        a pre-made phoenics configuration. if str, the name of a file from which to load it; Else, a dictionary.
        Individual keywords of the 'general' sections can also be passed down as kwargs
    save_to_file: bool:
        (Default value = False)
        whether or not to save the output of the optimization to an external file
    file_name: str:
        (Default value = None)
        where to save output to, if save_to_file is True.
    kwargs: dict:
        Send down more keywords for single replacements in the phoenics config 'general' section, like e.g. batches=5, boosted=True etc
    Returns
    -------

    """

    if variables is None:
        passives = None
    else:
        all_vars = Objective.extract_variables()
        passives = {}
        for k, v in initial_values.items():
            if k not in variables and k in all_vars:
                passives[k] = v
    optimizer = OptimizerPhoenics(samples=samples, backend=backend, maxiter=maxiter, silent=silent)
    return optimizer(objective=objective, backend=backend, passives=passives, previous=previous,
                     maxiter=maxiter, noise=noise, samples=samples,
                     phoenics_config=phoenics_config, save_to_file=save_to_file, file_name=file_name, *args, **kwargs)
