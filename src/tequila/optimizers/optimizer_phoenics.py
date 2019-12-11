from tequila.objective import Objective
from tequila.optimizers.optimizer_base import Optimizer
import multiprocessing as mp
import numpy as np
import copy
import warnings
import pickle
from tequila import TequilaException
warnings.simplefilter("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore")
    import phoenics
from numpy import pi as pi
import os
from collections import namedtuple
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
warnings.filterwarnings('ignore', category=FutureWarning)

OptimizerReturnType = namedtuple('OptimizerReturnType', 'energy angles history')

### wrapper for Phoenics, so that it can be used as an optimizer for parameters.
class PhoenicsOptimizer(Optimizer):

    def __init__(self, samples=None, simulator=None, save_history=True,minimize=True):
        self.minimize = minimize
        super().__init__(simulator=simulator, samples=samples, save_history=save_history)


    def _process_for_sim(self,recommendation):
        '''
        renders a set of recommendations usable by the QCircuit as a list of parameter sets to choose from.
        '''
        rec=copy.deepcopy(recommendation)
        for part in rec:
            for k,v in part.items():
                part[k]=v.item()

        return rec


    def _process_for_phoenics(self,pset, result):
        new=copy.deepcopy(pset)
        for k,v in new.items():
            new[k] = np.array([v],dtype=np.float32)
        new['E'] = result

        return new


    def _make_phoenics_object(self,objective,conf):
        if conf is not None:
            if hasattr(conf,'readlines'):
                bird=phoenics.Phoenics(config_file=conf)
            else:
                bird= phoenics.Phoenics(config_dict=conf)

            return bird
        op=objective.extract_variables()
        config={"general": {"auto_desc_gen": "False","batches":int(np.log2(mp.cpu_count())),"boosted":"False","parallel":"True"}}
        config['parameters']=[{'name':k, 'periodic':'True','type':'continuous','size':1,'low':0,'high':2*pi} for k in op.keys()]
        if self.minimize is True:
            config['objectives']=[{"name": "E", "goal": "minimize"}]
        else:
            config['objectives']=[{"name": "E", "goal": "maximize"}]

        bird=phoenics.Phoenics(config_dict=config)
        return bird

    def __call__(self, objective: Objective, maxiter: int =100, previous=None, phoenics_config=None, save_to_file=False, file_name=None):
        bird = self._make_phoenics_object(objective,phoenics_config)
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

        simulator = self.initialize_simulator(samples=self.samples)
        recompiled = []
        for u in objective.unitaries:
            recompiled.append(simulator.backend_handler.recompile(u))
        objective.unitaries =recompiled
        simulator.set_compile_flag(False)
        best=None
        best_angles=None
        O = objective
        u_num = mp.cpu_count()
        o_list = [copy.deepcopy(O) for i in range(u_num)]
        for i in range(maxiter):
            with warnings.catch_warnings():
                np.testing.suppress_warnings()
                warnings.simplefilter("ignore")
                warnings.filterwarnings("ignore",category=FutureWarning)
                if len(obs)>=1:
                    precs=bird.recommend(observations=obs)
                else:
                    precs=bird.recommend()

            pool=mp.Pool()
            runs=[]
            recs=self._process_for_sim(precs)
            for i,rec in enumerate(recs):
                running = o_list[i].update_variables(rec)
                if self.samples is None:
                    En=pool.apply_async(simulator.simulate_objective,[running])
                    # En = simulator.simulate_objective(objective=running)
                else:
                    En = pool.apply_async(simulator.measure_objective, [running, self.samples])
                    # En = simulator.measure_objective(objective=running, samples=self.samples)
                runs.append((rec, En))
            for run in runs:
                angles=run[0]
                E=run[1].get()
                #E=run[1]
                if best is None:
                    best=E
                    best_angles=angles
                else:
                    if self.minimize:
                        if E< best:
                            best=E
                            best_angles=rec
                        else:
                            pass
                    else:
                        if E> best:
                            best=E
                            best_angles=rec
                        else:
                            pass

                if self.save_history:
                    self.history.energies.append(E)
                    self.history.angles.append(angles)
                obs.append(self._process_for_phoenics(angles,E))

        if save_to_file is True:
            with open(file_name,'wb') as file:
                pickle.dump(obs,file)
        return OptimizerReturnType(energy=best, angles=best_angles, history=self.history)
