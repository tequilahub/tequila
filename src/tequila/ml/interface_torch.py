import torch
from .utils_ml import TequilaMLException, preamble
from tequila.objective import Objective
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

    comped_objective, compile_args, weight_vars, w_grads, i_grads, first, second \
        = preamble(objective, compile_args, input_vars)
    samples = compile_args['samples']

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
        def forward(ctx, input, angles):
            """
            forward pass of the function.
            """

            ctx.save_for_backward(input, angles)
            call_args = tensor_fix(input, angles, first, second)
            result = comped_objective(variables=call_args, samples=samples)
            if not isinstance(result, np.ndarray):
                # this happens if the Objective is a scalar since that's usually more convenient for pure quantum stuff.
                result = np.array(result)
            """
            for entry in input:
                if isinstance(entry, torch.Tensor):
                    if entry.is_cuda:
                        return torch.as_tensor(torch.from_numpy(result), dtype=input.dtype, device=entry.get_device())
            """
            r = torch.from_numpy(result)
            r.requires_grad_(True)
            return r

        @staticmethod
        def backward(ctx, grad_backward):
            input, angles = ctx.saved_tensors
            print('hey, it the back pass!')
            call_args = tensor_fix(input, angles, first, second)
            back_d = grad_backward.get_device()
            # build up weight and input gradient matrices... see what needs to be done to them.
            grad_outs = [None,None]
            for i, grads in enumerate([w_grads, i_grads]):
                print(i)
                if grads != {}:
                    g_keys = [j for j in grads.keys()]
                    probe = grads[g_keys[0]]  # first entry will tell us number of output
                    dims = len(g_keys), len(probe)
                    arr = np.empty(dims, dtype=np.float)
                    for j, key in enumerate(g_keys):
                        line = grads[key]
                        for k, ob in enumerate(line):
                            arr[j, k] = ob(variables=call_args, samples=samples)
                    if back_d >= 0:
                        g_tensor = torch.as_tensor(arr, dtype=grad_backward.dtype, device=back_d)
                    else:
                        g_tensor = torch.as_tensor(arr, dtype=grad_backward.dtype)
                    print('g_tensor', g_tensor)
                    print('grad_backward:', grad_backward)
                    jvp = torch.matmul(g_tensor, grad_backward)
                    jvp_out = jvp.flatten()
                    jvp_out.requires_grad_(True)
                    grad_outs[i] = jvp_out
            return tuple(grad_outs)

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
        self.weights = {}
        if inits is not None:
            for v in weight_vars:
                pv = torch.from_numpy(np.asarray(inits[v]))
                self.weights[str(v)] = torch.nn.Parameter(pv)
                self.register_parameter(str(v), self.weights[str(v)])
        else:
            for v in weight_vars:
                self.weights[str(v)] = torch.nn.Parameter(torch.nn.init.uniform(torch.Tensor(1),a=0.0,b=2*np.pi)[0])
                self.register_parameter(str(v), self.weights[str(v)])

    def forward(self, x=None):
        if x is not None:
            if len(x.shape) == 1:
                out = self._do(x)
            else:
                out = torch.stack([self._do(y) for y in x])
        else:
            out = self._do(None)
        out.requires_grad_(True)
        return out

    def _do(self, x):
        f = torch.stack([*self.parameters()])
        print(f)
        if x is not None:
            if len(x) != self._input_len:
                raise TequilaMLException('Received input of len {} when Objective takes {} inputs.'.format(len(x),self._input_len))
        return self.function.apply(x, f)


def tensor_fix(tensor,angles,first,second):
    """
    take a pytorch tensor and a dict of  int,Variable to create a variable,float dictionary therefrom.
    Parameters
    ----------
    tensor: torch.Tensor:
        a tensor.
    angles: torch.Tensor:
    first: dict:
        dict of int,Variable pairs indicating which position in Tensor corresponds to which variable.
    second: dict:
        dict of int,Variable pairs indicating which position in angles corresponds to which variable.
    Returns
    -------
    dict:
        dict of variable, float pairs. Can be used as call arg by underlying tq objectives
    """
    back = {}
    if tensor is not None:
        for i, val in enumerate(tensor):
            back[first[i]] = val.item()
    for i, val in enumerate(angles):
        back[second[i]] = val.item()
    return back
