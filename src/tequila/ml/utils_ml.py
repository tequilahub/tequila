from tequila.utils.exceptions import TequilaException
from tequila.objective.objective import assign_variable, Variable,FixedVariable,Objective
import typing
from tequila.simulators.simulator_api import compile
from tequila.circuit.gradient import grad


class TequilaMLException(TequilaException):
    pass


def check_compiler_args(c_args: dict) -> typing.Dict:
    """
    Check if a dictionary's keys are exclusively arguments accepted by tq.compile.
    If the dictionary is correct, it is returned.
    Parameters
    ----------
    c_args: dict:
        a dictionary whose validity as kwargs for compilation is to be established.

    Returns
    -------
    dict:
        a dictionary that can serve as kwargs for tq.compile

    See Also
    --------
    tequila.simulators.simulator_api.compile
    """

    valid_keys=['backend','samples','noise','device','initial_values']
    if c_args is None:
        return {k:None for k in valid_keys}
    for k in c_args.keys():
        if k not in valid_keys:
            raise TequilaException('improper keyword {} found in compilation kwargs dict; please try again.'.format(k))
        else:
            pass
    return c_args


def check_full_span(all_vars: list, combined: list):
    """
    check that two lists of variables have all the same elements, even if in different orders.

    helper function for use in the ml preamble to ensure that the formatted input and weight variables combine
    to give all the variables of the objective.

    Parameters
    ----------
    all_vars: list:
        a list of Tequila Variables, interpreted contextually as all the variables extract from an objective.
    combined: list:
        a list of Tequila Variables, interpreted contextually as the weight and input variable lists combined.

    """
    s1 = set(all_vars)
    s2 = set(combined)
    if not s1 == s2:
        raise TequilaException('Variables of the Objective and the input-and-weights lists are not identical!')


def preamble(objective: Objective,compile_args: dict = None,input_vars: list = None):
    """
    Helper function for use at the beggining of
    Parameters
    ----------
    objective: Objective:
        the objective to manipulate and compile.
    compile_args: dict, optional:
        a dictionary of args that can be passed as kwargs to tq.compile
    input_vars: list, optional:
        a list of variables of the objective to specify as input, rather than itnernal weights.

    Returns
    -------
    tuple
        the compiled objective, a list of input_variables, and a list of weight_variables.
    """
    all_vars = objective.extract_variables()
    compile_args = check_compiler_args(compile_args)

    weight_vars=[]
    if input_vars is None:
        input_vars = []
        weight_vars = all_vars
    else:
        input_vars=[assign_variable(v) for v in input_vars]
        for var in all_vars:
            if var not in input_vars:
                weight_vars.append(assign_variable(var))

    check_full_span(all_vars,input_vars.extend(weight_vars))
    initvals = compile_args['initial_values']
    if initvals is not None:
        for k in initvals.keys():
            if assign_variable(k) in input_vars:
                raise TequilaMLException('initial_values contained key {}, which is meant to be an input variable.'.format(k))
    comped = compile(objective,**compile_args)
    return comped,compile_args,weight_vars,input_vars


def get_gradients(objective: Objective, compile_args: dict):
    """
    get the gradients of the Objective and compile them all.
    Parameters
    ----------
    objective: Objective:
        an Objective
    compile_args: dict:
        compilation arguments for compiling the gradient once it is constructed.

    Returns
    -------
    Dict:
        the gradient, compiled for use.

    """
    compile_args = check_compiler_args(compile_args)
    grads=grad(objective)
    back = {}
    for k,v in enumerate(grads):
        new = []
        for o in v:
            new.append(compile(o,**compile_args))
        back[k] = new

    return back


def separate_gradients(gradients,weight_vars, input_vars):
    """
    split the gradient dictionary up into one for input and one for weights.

    Parameters
    ----------
    gradients: dict:
        dictionary of gradients. Should be str, list[Objective] key/value pairs.
    input_vars: list:
        the input variables of the Objective
    weight_vars: list:
        the internal weight variables of the Objective

    Returns
    -------
    tuple(dict):
        a pair of dictionaries, respectively the gradients for input and weight variables.
    """
    i_grad = {}
    w_grad = {}

    for var in input_vars:
        i_grad[var] = gradients[var]
    for var in weight_vars:
        w_grad[var] = gradients[var]
    return w_grad, i_grad

def get_variable_orders(weight_vars,input_vars):
    """
    get a dictionary mapping position in an array to tequila variables.
    Parameters
    ----------
    input_vars: list:
        which vars are interpreted as input
    weight_vars: list:
        which vars are interpreted as weight

    Returns
    -------

    """
    displace=0
    pattern = {}
    for i,v in enumerate(weight_vars):
        displace += 1
        pattern[i] = v
    for j,v in enumerate(input_vars):
        pattern[j + displace] = v

    return pattern