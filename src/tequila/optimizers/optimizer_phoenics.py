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
        return ["phoenics"]

    def __init__(self, maxiter, backend=None, save_history=True, minimize=True,
                 samples=None, silent=None, noise=None, device=None):
        self._minimize = minimize

        super().__init__(backend=backend, maxiter=maxiter, samples=samples,
                         noise=noise,device=device,
                         save_history=save_history, silent=silent)

    def _process_for_sim(self, recommendation, passive_angles):
        '''
        renders a set of recommendations usable by the QCircuit as a list of parameter sets to choose from.
        '''
        rec = copy.deepcopy(recommendation)
        for part in rec:
            for k, v in part.items():
                part[k] = v.item()
            if passive_angles is not None:
                for k, v in passive_angles.items():
                    part[k] = v
        return rec

    def _process_for_phoenics(self, pset, result, passive_angles=None):
        new = copy.deepcopy(pset)
        for k, v in new.items():
            new[k] = np.array([v], dtype=np.float32)
        if passive_angles is not None:
            for k in passive_angles.keys():
                del new[k]
        new['Energy'] = result

        return new

    def _make_phoenics_object(self, objective, passive_angles=None, conf=None, *args, **kwargs):
        if conf is not None:
            if hasattr(conf, 'readlines'):
                bird = phoenics.Phoenics(config_file=conf)
            else:
                bird = phoenics.Phoenics(config_dict=conf)

            return bird
        op = objective.extract_variables()
        if passive_angles is not None:
            for i, thing in enumerate(op):
                if thing in passive_angles.keys():
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
                 maxiter=None,
                 variables: typing.List[Variable] = None,
                 initial_values: typing.Dict[Variable, numbers.Real] = None,
                 previous=None,
                 phoenics_config=None,
                 save_to_file=False,
                 file_name=None,
                 *args,
                 **kwargs):

        active_angles, passive_angles, variables = self.initialize_variables(objective,
                                                               initial_values=initial_values,
                                                               variables=variables)

        if maxiter is None:
            maxiter = 10

        obs = []
        bird = self._make_phoenics_object(objective, passive_angles, phoenics_config, *args, **kwargs)
        if previous is not None:
            if type(previous) is str:
                try:
                    obs = pickle.load(open(previous, 'rb'))
                except:
                    print(
                        'failed to load previous observations, which are meant to be a pickle file. Starting fresh.')
            elif type(previous) is list:
                if all([type(k) == dict for k in previous]):
                    obs = previous
                else:
                    print('previous observations were not in the correct format (list of dicts). Starting fresh.')



        if not (type(file_name) == str or file_name == None):
            raise TequilaException('file_name must be a string, or None!')

        best = None
        best_angles = None

        # avoid multiple compilations
        compiled_objective = compile_objective(objective=objective, backend=self.backend,
                                               device=self.device,
                                               samples=self.samples, noise=self.noise)

        if not self.silent:
            print('phoenics has recieved')
            print("objective: \n")
            print(objective)
            print("noise model : {}".format(self.noise))
            print("samples     : {}".format(self.samples))
            print("maxiter     : {}".format(maxiter))
            print("variables   : {}".format(objective.extract_variables()))
            print("passive var : {}".format(passive_angles))
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
            recs = self._process_for_sim(precs, passive_angles=passive_angles)

            start = time.time()
            for j, rec in enumerate(recs):
                En = compiled_objective(variables=rec, samples=self.samples, noise=self.noise)
                runs.append((rec, En))
                if not self.silent:
                    if self.print_level > 2:
                        print("energy = {:+2.8f} , angles=".format(En), rec)
                    else:
                        print("energy = {:+2.8f}".format(En))
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
                obs.append(self._process_for_phoenics(angles, E, passive_angles=passive_angles))

        if file_name is not None:
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
             device: str = None,
             previous: typing.Union[str, list] = None,
             phoenics_config: typing.Union[str, typing.Dict] = None,
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
    device: str:
        (Default value = None)
        the device from which to (potentially, simulatedly) sample all quantum circuits employed in optimization.
    previous:
        (Default value = None)
        Previous phoenics observations. If string, the name of a file from which to load them. Else, a list.
    phoenics_config:
        (Default value = None)
        a pre-made phoenics configuration. if str, the name of a file from which to load it; Else, a dictionary.
        Individual keywords of the 'general' sections can also be passed down as kwargs
    file_name: str:
        (Default value = None)
        where to save output to, if save_to_file is True.
    kwargs: dict:
        Send down more keywords for single replacements in the phoenics config 'general' section, like e.g. batches=5, boosted=True etc
    Returns
    -------

    """

    optimizer = OptimizerPhoenics(samples=samples, backend=backend,
                                  noise=noise,device=device,
                                  maxiter=maxiter, silent=silent)
    return optimizer(objective=objective, initial_values=initial_values, variables=variables, previous=previous,
                     maxiter=maxiter,
                     phoenics_config=phoenics_config, file_name=file_name, *args, **kwargs)
