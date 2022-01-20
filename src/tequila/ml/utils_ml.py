from tequila.utils.exceptions import TequilaException
from tequila.objective.objective import assign_variable, Objective,\
    format_variable_dictionary, format_variable_list
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

    valid_keys=['backend', 'samples', 'noise', 'device', 'initial_values']
    if c_args is None:
        return {k:None for k in valid_keys}
    for k in c_args.keys():
        if k not in valid_keys:
            raise TequilaException('improper keyword {} found in compilation kwargs dict; please try again.'.format(k))
        else:
            pass
    for k in valid_keys:
        if k in c_args.keys():
            pass
        else:
            c_args[k] = None

    return c_args


def preamble(objective: Objective, compile_args: dict = None, input_vars: list = None):
    """
    Helper function for interfaces to ml backends.
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
        the compiled objective, it's compile arguments, its weight variables, dicts for the weight and input gradients,
        and a dictionary that links positions in an array to each variable (parses parameters).
    """
    def var_sorter(e):
        return hash(e.name)
    all_vars = objective.extract_variables()
    all_vars.sort(key=var_sorter)
    compile_args = check_compiler_args(compile_args)

    weight_vars = []
    if input_vars is None:
        input_vars = []
        weight_vars = all_vars
    else:
        input_vars = [assign_variable(v) for v in input_vars]
        for var in all_vars:
            if var not in input_vars:
                weight_vars.append(assign_variable(var))

    init_vals = compile_args['initial_values']
    if init_vals is not None:
        for k in init_vals.keys():
            if assign_variable(k) in input_vars:
                raise TequilaMLException('initial_values contained key {},'
                                         'which is meant to be an input variable.'.format(k))
        init_vals = format_variable_dictionary(init_vals)
    compile_args.pop('initial_values')

    comped = compile(objective, **compile_args)

    gradients = get_gradients(objective, compile_args)
    w_grad, i_grad = separate_gradients(gradients, weight_vars=weight_vars, input_vars=input_vars)
    first, second = get_variable_orders(weight_vars, input_vars)
    compile_args['initial_values']=init_vals
    return comped, compile_args, input_vars, weight_vars, i_grad, w_grad, first, second


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
    grads = grad(objective)
    back = {}
    for k, v in grads.items():
        new = []
        if isinstance(v, Objective):
            new.append(compile(v, **compile_args))
        else:
            for o in v:
                new.append(compile(o, **compile_args))
        back[k] = new

    return back


def separate_gradients(gradients, weight_vars, input_vars):
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


def get_variable_orders(weight_vars, input_vars):
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
    tuple
        dicts, which position in tensor of input and tensor of parameters corresponds to which variable.
    """
    first = {}
    second = {}
    for i, v in enumerate(input_vars):
        first[i] = v
    for j, v in enumerate(weight_vars):
        second[j] = v
    return first, second

