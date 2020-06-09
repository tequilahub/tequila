import torch
from .utils_ml import TequilaMLException, preamble, get_gradients, separate_gradients, get_variable_orders
from tequila.objective import Objective, Variable
import numpy as np


def get_torch_function(objective: Objective, compile_args: dict = None, input_vars: list = None):
    """
    build a torch autograd function that calls the Objective; return it, and other useful objects.

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
        """
        Internal class for the forward and backward passes of calling a tequila objective.

        Notes
        -----

        Though this class is a private class, some explanations of it's implementation may benefit
        curious users and later developers.

        Question: why is this class defined within a function?
        answer: because, since its defined entirely with staticmethods -- about which torch is quite particular --
        it is impossible to use the attribute 'self' to store information for later use.
        This means that, to wrap arbitrary tequila Objectives into this class, the class needs to see the in some
        kind of well contained scope; the function containing this class, get_torch_function, provides that scope.
        in particular, this scoping is used to associate, with this function, arbitrary tequila objectives,
        their gradients with respect to weights or inputs (that is, variables specified to be one or the other)
        and a small dictionary called pattern, which orders the tequila variables w.r.t the order in which a tensor
        of combined input values and weight values are passed down to the function.

        Though this class doesn't have any proper attributes seperate from those it inherits, we detail
        the non-torch objects called within the function here:

        For Forward
        comped_objective: Objective
            a compiled tequila objective; this function has merely wrapped around it to pass torch Tensors into it.


        For Forward and Backward
        samples: int or None:
            how many samples the user wants when sampling the Objective or it's gradients.


        methods called:
            tensor_fix:
                takes a tensor and an (int: Variable) dict and returns a (Variable: float) dict.

        """
        @staticmethod
        def forward(ctx, input):
            """
            forward pass of the function.
            """
            ctx.call_args = tensor_fix(input, pattern)
            ctx.save_for_backward(input)
            result = comped_objective(variables=ctx.call_args,samples=samples)
            if not isinstance(result, np.ndarray):
                # this happens if the Objective is a scalar since that's usually more convenient for pure quantum stuff.
                result = np.array(result)

            for entry in input:
                if isinstance(entry, torch.Tensor):
                    if entry.is_cuda:
                        return torch.as_tensor(torch.from_numpy(result),dtype=input.dtype, device=entry.get_device())

            r = torch.from_numpy(result)
            return r

        @staticmethod
        def backward(ctx, grad_backward):
            call_args = ctx.call_args
            back_d =grad_backward.get_device()
            # build up weight and input gradient matrices... see what needs to be done to them.
            if w_grads != {}:
                # this calculate the gradient w.r.t weights of this layer
                w_keys = [j for j in w_grads.keys()]
                w_probe = w_grads[w_keys[0]]
                w_dims = len(w_keys), len(w_probe)
                w_array = np.empty(w_dims, dtype=np.float)
                for i, key in enumerate(w_keys):
                    line = w_grads[key]
                    for j, ob in enumerate(line):
                        w_array[i, j] = ob(variables=call_args,samples=samples)
                if back_d >= 0:
                    w_tensor = torch.as_tensor(w_array, dtype=grad_backward.dtype,device=back_d)
                else:
                    w_tensor = torch.as_tensor(w_array, dtype=grad_backward.dtype)
                w_jvp = torch.matmul(w_tensor, grad_backward)
                w_out = w_jvp.flatten()
                w_out.requires_grad_(True)
            else:
                w_out = None

            if i_grads != {}:
                # same as above, since this is quantum layer; get the gradient w.r.t network input
                i_keys = [k for k in i_grads.keys()]
                i_probe = i_grads[i_keys[0]]
                i_dims = len(i_keys), len(i_probe)
                i_array = np.empty(i_dims, dtype=np.float)
                for i, key in enumerate(i_keys):
                    line = i_grads[key]
                    for j, ob in enumerate(line):
                        i_array[i, j] = ob(variables=call_args,samples=samples)
                if back_d >= 0:
                    i_tensor = torch.as_tensor(i_array, dtype=grad_backward.dtype,device=back_d)
                else:
                    i_tensor = torch.as_tensor(i_array,dtype=grad_backward.dtype)
                i_jvp = torch.matmul(i_tensor, grad_backward)
                i_out = i_jvp.flatten()
                i_out.requires_grad_(True)
            else:
                i_out = None
            return w_out, i_out

    return _TorchFunction, weight_vars, compile_args


class TorchLayer(torch.nn.Module):
    """
    class representing a tequila Objective wrapped for use by pytorch.
    """

    def __init__(self, objective, compile_args, input_vars):
        super().__init__()

        self._objective = objective
        self.function,  weight_vars, compile_args = get_torch_function(objective, compile_args, input_vars)
        self._input_len = len(objective.extract_variables()) - len(weight_vars)
        inits = compile_args['initial_values']
        self.weights={}
        if inits is not None:
            for v in weight_vars:
                pv = torch.from_numpy(np.asarray(inits[v]))
                self.weights[str(v)] = torch.nn.Parameter(pv)
                self.register_parameter(str(v), self.weights[str(v)])
        else:
            for v in weight_vars:
                self.weights[str(v)] = torch.nn.Parameter(torch.nn.init.uniform(torch.Tensor(1),a=0.0,b=2*np.pi)[0])
                self.register_parameter(str(v), self.weights[str(v)])

    def forward(self,input = None):
        weights =[]
        for v in self.weights.values():
            weights.append(v.detach())
        cat = torch.stack([p for p in self.parameters()])
        if input is not None:
            send = torch.stack([cat,input])
        else:
            send = cat
        out = self.function.apply(send)
        out.requires_grad_(True)
        return out


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
