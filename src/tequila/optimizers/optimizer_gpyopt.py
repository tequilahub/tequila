from tequila.objective.objective import Objective
from tequila.optimizers.optimizer_base import Optimizer
import typing
import numbers
from tequila.objective.objective import assign_variable, Variable, format_variable_dictionary, format_variable_list
import warnings
warnings.simplefilter("ignore")

__HAS_GPYOPT__ = False
try:
    import GPyOpt
    from GPyOpt.methods import BayesianOptimization
    __HAS_GPYOPT__ = True
except:
    __HAS_GPYOPT__ = False


import numpy as np
from tequila.simulators.simulator_api import compile, simulate
from collections import namedtuple
GPyOptReturnType = namedtuple('GPyOptReturnType', 'energy angles history opt')




def array_to_objective_dict(objective,array,passives=None) -> typing.Dict[Variable,float]:
    op = objective.extract_variables()
    if passives is not None:
        for i, thing in enumerate(op):
            if thing in passives.keys():
                op.remove(thing)
    back={v:array[:,i] for i,v in enumerate(op)}
    if passives is not None:
        for k,v in passives.items():
            back[k]=v
    return back


class GPyOptOptimizer(Optimizer):

    def __init__(self, maxiter=100,backend=None, save_history=True,minimize=True,samples=None):
        self._minimize = minimize
        super().__init__(simulator=backend,maxiter=maxiter, samples=samples, save_history=save_history)

    def get_domain(self,objective,passives=None) -> typing.List[typing.Dict]:
        op=objective.extract_variables()
        if passives is not None:
            for i, thing in enumerate(op):
                if thing in passives.keys():
                    op.remove(thing)
        return [{'name':v,'type':'continuous','domain':(0,2*np.pi)} for v in op]

    def get_object(self,func,domain,method) -> GPyOpt.methods.BayesianOptimization:
        return BayesianOptimization(f=func,domain=domain,acquisition=method)

    def construct_function(self,objective,backend,passives=None,samples=None,noise_model=None) -> typing.Callable:
        return lambda arr: objective(backend=backend,
                                    variables=array_to_objective_dict(objective,arr,passives),
                                    samples=samples,
                                    noise_model=noise_model)

    def redictify(self,arr,objective,passives=None) -> typing.Dict:
        op=objective.extract_variables()
        if passives is not None:
            for i, thing in enumerate(op):
                if thing in passives.keys():
                    op.remove(thing)
        back={v:arr[i] for i,v in enumerate(op)}
        if passives is not None:
            for k, v in passives.items():
                back[k] = v
        return back

    def __call__(self, objective: Objective,
                 maxiter: int,
                 passives: typing.Dict[Variable,numbers.Real] = None,
                 samples: int = None,
                 backend: str = None,
                 noise = None,
                 method: str = 'lbfgs') -> GPyOptReturnType :
        if self.samples is not None:
            if samples is None:
                samples=self.samples
            else:
                pass
        else:
            pass
        dom=self.get_domain(objective,passives)
        init={v:np.random.uniform(0,2*np.pi) for v in objective.extract_variables()}
        ### O is broken, not using it right now
        O= compile(objective=objective,variables=init, backend=backend,noise_model=noise, samples=samples)
        f = self.construct_function(O,backend,passives,samples,noise_model=noise)
        opt=self.get_object(f,dom,method)
        opt.run_optimization(maxiter)
        if self.save_history:
            self.history.energies=opt.get_evaluations()[1].flatten()
            self.history.angles=[self.redictify(v,objective,passives) for v in opt.get_evaluations()[0]]
        return GPyOptReturnType(energy=opt.fx_opt, angles=self.redictify(opt.x_opt,objective,passives),
                                history=self.history, opt=opt)

def minimize(objective: Objective,
             maxiter: int,
             samples: int = None,
             variables: typing.List=None,
             initial_values: typing.Dict=None,
             backend: str = None,
             noise =None,
             method: str= 'lbfgs'
             ) -> GPyOptReturnType :

    """

    Parameters
    ----------
    objective: Objective :
        The tequila objective to optimize
    initial_values: typing.Dict[typing.Hashable, numbers.Real]: (Default value = None):
        Initial values as dictionary of Hashable types (variable keys) and floating point numbers. generates FIXED variables! if not provided,
        all variables will be optimized.
    variables: typing.List[typing.Hashable] :
         (Default value = None)
         List of Variables to optimize. If None, all variables optimized, and the passives command is over-ruled.
    samples: int :
         (Default value = None)
         samples/shots to take in every run of the quantum circuits (None activates full wavefunction simulation)
    maxiter: int :
         how many iterations of GPyOpt to run. Note: GPyOpt will override this as it sees fit.
    backend: str :
         (Default value = None)
         Simulator backend, will be automatically chosen if set to None
    noise: NoiseModel :
         (Default value = None)
        a noise model to apply to the circuits of Objective.
    method: str:
         (Default value = 'lbfgs')
         method of acquisition. Allowed arguments are 'lbfgs', 'DIRECT', and 'CMA'

    Returns
    -------

    """



    if variables is None:
        passives=None
    else:
        all_vars = Objective.extract_variables()
        passives = {}
        for k,v in initial_values.items():
            if k not in variables and k in all_vars:
                passives[k]=v
    optimizer=GPyOptOptimizer()
    return optimizer(objective=objective,samples=samples,backend=backend,passives=passives,maxiter=maxiter,noise=noise,
                     method=method
                         )

