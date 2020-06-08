import torch
from .utils_ml import TequilaMLException,preamble,get_gradients,separate_gradients
from tequila.objective import Objective,Variable

def to_torch(objective:Objective,compile_args: dict = None, input_vars: list = None, weight_vars: list = None):
    """
    create a torch layer out of a Tequila objective.

    Parameters
    ----------
    objective: Objective:
        the Objective to be transformed into a torch layer.
    compile_args: dict:
        a dictionary of arguments for the tequila compiler; used to render objectives callable
    input_vars: list:
        a list of variables; indicates which variables' values will be considered input to the layer.
    weight_vars: list:
        a list of variables; indiciates which variables' values should be stored internally as weights.

    Returns
    -------
    type:
        a pytorch layer equivalent to a tequila objective.
    """

    comped_objective,input_vars,weight_vars = preamble(objective,compile_args,input_vars,weight_vars)
    gradients = get_gradients(objective,compile_args)
    i_grads, w_grads = separate_gradients(gradients,input_vars,weight_vars)

    raise TequilaMLException('Nothing implemented beyond this point. Function is not ready')

