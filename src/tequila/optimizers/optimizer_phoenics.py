from tequila.objective import Objective
from tequila.optimizers.optimizer_base import Optimizer
import typing
import numbers
from tequila.objective.objective import assign_variable, Variable, format_variable_dictionary, format_variable_list
import multiprocessing as mp
import copy
import warnings
import pickle
from tequila import TequilaException
warnings.simplefilter("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore")
    import phoenics
from tequila.autograd_imports import jax
import numpy as np
from numpy import pi as pi
from tequila.simulators import compile_objective, simulate_objective,sample_objective
import os
from collections import namedtuple
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
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
class PhoenicsOptimizer(Optimizer):

    def __init__(self, maxiter,backend=None, save_history=True,minimize=True,samples=None):
        self._minimize = minimize
        if samples is not None:
            print('warning you: the samples you input do not matter, except when calling')
        super().__init__(simulator=backend,maxiter=maxiter, samples=None, save_history=save_history)


    def _process_for_sim(self,recommendation,passives):
        '''
        renders a set of recommendations usable by the QCircuit as a list of parameter sets to choose from.
        '''
        rec=copy.deepcopy(recommendation)
        for part in rec:
            for k,v in part.items():
                part[k]=v.item()
            if passives is not None:
                for k,v in passives.items():
                    part[k]=v
        return rec


    def _process_for_phoenics(self,pset,result,passives=None):
        new=copy.deepcopy(pset)
        for k,v in new.items():
            new[k] = np.array([v],dtype=np.float32)
        if passives is not None:
            for k in passives.keys():
                del new[k]
        new['Energy'] = result

        return new


    def _make_phoenics_object(self,objective,passives=None,conf=None):
        if conf is not None:
            if hasattr(conf,'readlines'):
                bird=phoenics.Phoenics(config_file=conf)
            else:
                bird= phoenics.Phoenics(config_dict=conf)

            return bird
        op=objective.extract_variables()
        if passives is not None:
            for i, thing in enumerate(op):
                if thing in passives.keys():
                    del op[i]

        config={"general": {"auto_desc_gen": "False","batches":int(np.log2(mp.cpu_count())),"boosted":"False","parallel":"True"}}
        config['parameters']=[{'name':k, 'periodic':'True','type':'continuous','size':1,'low':0,'high':2*pi} for k in op]
        if self._minimize is True:
            config['objectives']=[{"name": "Energy", "goal": "minimize"}]
        else:
            config['objectives']=[{"name": "Energy", "goal": "maximize"}]

        bird=phoenics.Phoenics(config_dict=config)
        return bird

    def __call__(self, objective: Objective,
                 maxiter: int,
                 passives: typing.Dict[Variable,numbers.Real] = None,
                 samples: int = None,
                 backend: str = None,
                 previous=None,
                 phoenics_config=None,
                 save_to_file=False,
                 file_name=None):

        bird = self._make_phoenics_object(objective,passives,phoenics_config)
        if previous is not None:
            if type(previous) is str:
                try:
                    obs=pickle.load(open(previous, 'rb'))
                except:
                    print('failed to load previous observations, which are meant to be a pickle file. Please try again or seek assistance. Starting fresh.')
                    obs=[]
            elif type(previous) is list:
                if all([type(k)==dict for k in previous]):
                    obs=previous
                else:
                    print('previous observations were not in the correct format (list of dicts). Are you sure you gave me the right info? Starting fresh.')
                    obs=[]

        else:
            obs=[]

        if save_to_file is True:
            if type(file_name) is str:
                pass
            elif file_name is None:
                raise TequilaException('You have asked me to save phoenics observations without telling me where to do so! please provide a file_name')
            else:
                raise TequilaException('file_name must be a string!')

        ### this line below just gets the damn compiler to run, since that argument is necessary
        init = {key:np.pi for key in objective.extract_variables()}

        O= compile_objective(objective=objective,variables=init, backend=backend,
                                               samples=samples)

        best=None
        best_angles=None

        for i in range(0,maxiter):
            with warnings.catch_warnings():
                np.testing.suppress_warnings()
                warnings.simplefilter("ignore")
                warnings.filterwarnings("ignore",category=FutureWarning)
                blockPrint()
                if len(obs)>=1:

                    precs=bird.recommend(observations=obs)
                else:
                    precs=bird.recommend()
                enablePrint()
            runs=[]
            recs=self._process_for_sim(precs,passives=passives)
            for i,rec in enumerate(recs):
                if samples is None:
                    En = simulate_objective(objective=O,variables=rec)
                else:
                    En = sample_objective(objective=O,variables=rec, samples=samples)
                runs.append((rec, En))
            for run in runs:
                angles=run[0]
                E=run[1]
                if best is None:
                    best=E
                    best_angles=angles
                else:
                    if self._minimize:
                        if E< best:
                            best=E
                            best_angles=angles
                        else:
                            pass
                    else:
                        if E> best:
                            best=E
                            best_angles=angles
                        else:
                            pass

                if self.save_history:
                    self.history.energies.append(E)
                    self.history.angles.append(angles)
                obs.append(self._process_for_phoenics(angles,E,passives=passives))

        if save_to_file is True:
            with open(file_name,'wb') as file:
                pickle.dump(obs,file)
        return PhoenicsReturnType(energy=best, angles=best_angles, history=self.history, observations=obs)

def minimize(objective: Objective,
             maxiter: int,
             samples: int = None,
             variables: typing.List=None,
             initial_values: typing.Dict=None,
             backend: str = None,
             previous=None,
             phoenics_config=None,
             save_to_file=False,
             file_name=None):
    if variables is None:
        passives=None
    else:
        all_vars = Objective.extract_variables()
        passives = {}
        for k,v in initial_values.items():
            if k not in variables and k in all_vars:
                passives[k]=v
    optimizer=PhoenicsOptimizer(samples=samples,backend=backend,maxiter=maxiter)
    return optimizer(objective=objective,passives=passives,previous=previous,maxiter=maxiter,
                         phoenics_config=phoenics_config,save_to_file=save_to_file,file_name=file_name)

