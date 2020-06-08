import torch
from .utils_ml import TequilaMLException, preamble, get_gradients, separate_gradients, get_variable_orders
from tequila.objective import Objective, Variable
import numpy as np


def get_torch_function(objective: Objective, compile_args: dict = None, input_vars: list = None):
    """
    create a callable autograd function with forward and backward properly set for a given Objective.

    Parameters
    ----------
    objective: Objective:
        the Objective to be transformed into a torch layer.
    compile_args: dict:
        a dictionary of arguments for the tequila compiler; used to render objectives callable
    input_vars: list:
        a list of variables; indicates which variables' values will be considered input to the layer.

    Returns
    -------
    tuple:
        the requisite pytorch autograd function, alongside necessary information for higher level classes.
    """

    comped_objective, compile_args, weight_vars, input_vars = preamble(objective, compile_args, input_vars)
    samples = compile_args['samples']
    gradients = get_gradients(objective, compile_args)
    w_grads, i_grads = separate_gradients(gradients, input_vars=input_vars, weight_vars=weight_vars)
    pattern = get_variable_orders(weight_vars=weight_vars, input_vars=input_vars)

    class _TorchFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            """
            forward pass of the function.
            """
            ctx.call_args = tensor_fix(input, pattern)
            ctx.save_for_backward(input)

            result = comped_objective(variables=ctx.call_args,samples=samples)
            if not isinstance(result, np.ndarray):
                result = np.array(result)

            for entry in input:
                if isinstance(entry, torch.Tensor):
                    if entry.is_cuda:
                        return torch.as_tensor(torch.from_numpy(result), device=entry.get_device())

            return torch.from_numpy(result)

        @staticmethod
        def backward(ctx, grad_backward):
            """
            Backward pass of the function.
            """
            call_args = ctx.call_args

            # build up weight and input gradient matrices... see what needs to be done to them.
            if w_grads != {}:

                # we need to figure out the dimension of the weight jacobian
                w_keys = w_grads.keys()
                w_probe = w_grads[w_keys[0]]
                w_dims = len(w_keys), len(w_probe)
                w_array = np.empty(w_dims, dtype=np.float)
                for i, key in enumerate(w_keys):
                    line = w_grads[key]
                    for j, ob in enumerate(line):
                        w_array[i, j] = line(variables=call_args,samples=samples)
                w_tensor = torch.as_tensor(w_array, dtype=grad_backward.dtype)
                w_jvp = torch.matmul(w_tensor, grad_backward)
                w_out = w_jvp.flatten()
            else:
                w_out = None

            if i_grads != {}:
                i_keys = i_grads.keys()
                i_probe = i_grads[i_keys[0]]
                i_dims = len(i_keys), len(i_probe)
                i_array = np.empty(i_dims, dtype=np.float)
                for i, key in enumerate(i_keys):
                    line = i_grads[key]
                    for j, ob in enumerate(line):
                        i_array[i, j] = line(variables=call_args,samples=samples)

                i_tensor = torch.as_tensor(i_array, dtype=grad_backward.dtype)
                i_jvp = torch.matmul(i_tensor, grad_backward)
                w_out = i_jvp.flatten()
            else:
                i_out = None

            return w_out, i_out

    return _TorchFunction(), weight_vars, compile_args


class TorchLayer(torch.nn.Module):
    """
    class representing a tequila Objective wrapped for use by pytorch.
    """

    def __init__(self, objective, compile_args, input_vars):
        super().__init__()

        self.function, compile_args, weight_vars = get_torch_function(objective, compile_args, input_vars)
        inits=compile_args['initial_values']
        self.weights={}
        if inits is not None:
            for v in weight_vars:
                pv = torch.from_numpy(np.asarray(inits[v]))
                self.weights[str(v)] = torch.nn.Parameter(pv[0])
        else:
            for v in weight_vars:
                self.weights[str(v)] = torch.nn.Parameter(torch.nn.init.uniform(torch.Tensor(1),a=0.0,b=2*np.pi)[0])



    def forward(self, input):
        weights =[]
        for v in self.weights.values():
            weights.append(v.detach())
        cat = torch.cat(weights)
        send = torch.cat([cat,input])
        return self.function(input)


def tensor_fix(tensor, pattern):
    """
    take a pytorch tensor and a dict of  int,Variable to create a variable,float dictionary therefrom.
    Parameters
    ----------
    tensor: torch.Tensor:
        a tensor.
    pattern: dict:
        dict of int,Variable pairs indicating which position in Tensor corresponds to which variable.

    Returns
    -------
    dict:
        dict of variable, float pairs. Can be used as call arg by underlying tq objectives
    """
    back = {}
    for i, val in enumerate(tensor):
        back[pattern[i]] = val.item()
    return back
