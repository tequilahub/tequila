from collections import OrderedDict
from typing import List

from typing import Union, Dict, Callable, Any

from tequila.ml.utils_ml import preamble, TequilaMLException
from tequila.objective import Objective, VectorObjective, Variable, vectorize
from tequila.tools import list_assignment
from tequila.simulators.simulator_api import simulate
import numpy as np

import tensorflow as tf

class TFLayer(tf.keras.layers.Layer):
    """
    Tensorflow Layer

    DISCLAIMER:
    This is very much a WIP, since we are not exactly sure how users intend to use it. Please feel free to raise issues
    and give feedback without hesitation.
    """
    def __init__(self, objective: Union[Objective, VectorObjective], compile_args: Dict[str, Any] = None,
                 input_vars: Dict[str, Any] = None, **kwargs):
        """
        Tensorflow layer that compiles the Objective (or VectorObjective) with the given compile arguments and/or
        input variables if there are any when initialized. When called, it will forward the input variables into the
        compiled objective (if there are any inputs needed) alongside the parameters and will return the output.
        The gradient values can also be returned.

        Parameters
        ----------
        objective
            Objective or VectorObjective to compile and run.
        compile_args
            dict of all the necessary information to compile the objective
        input_vars
            List of variables that will be inputs
        """
        super(TFLayer, self).__init__(**kwargs)

        # Currently, the optimizers in tf.keras.optimizers don't support float64. For now, all values will be cast to
        # float32 to accommodate this, but in the future, whenever it is supported, this can be changed with
        # set_cast_type()
        self._cast_type = tf.float32

        self.objective = objective
        # Store the objective and vectorize it if necessary
        if isinstance(objective, tuple) or isinstance(objective, list):
            for i, elem in enumerate(objective):
                if not isinstance(elem, Objective):
                    raise TequilaMLException("Element {} in {} is not a Tequila Objective: {}"
                                             "".format(i, type(objective), elem))
            objective = vectorize(list_assignment(objective))

        elif isinstance(objective, Objective) or isinstance(objective, VectorObjective):
            objective = vectorize(list_assignment(objective))
        else:
            raise TequilaMLException("Objective must be a Tequila Objective, VectorObjective "
                                     "or list/tuple of Objectives. Received a {}".format(type(objective)))
        self.objective = objective

        # Compile the objective, prepare the gradients and whatever else that may be necessary
        self.comped_objective, self.compile_args, self.input_vars, self.weight_vars, self.i_grads, self.w_grads, \
        self.first, self.second = preamble(objective, compile_args, input_vars)

        # VARIABLES
        # These variables will hold 1D tensors which each will store the values in the order found by self.input_vars
        # for the variable in self.input_variable, and in the order found by self.weight_vars for the variable in
        # self.weight_variable

        # If there are inputs, prepare an input tensor as a trainable variable
        # NOTE: if the user specifies values for the inputs, they will be assigned in the set_input_values()
        if self.input_vars:
            initializer = tf.constant_initializer(np.random.uniform(low=0., high=2 * np.pi, size=len(self.input_vars)))
            self.input_variable = self.add_weight(name="input_tensor_variable",
                                                  shape=(len(self.input_vars)),
                                                  dtype=self._cast_type,
                                                  initializer=initializer,
                                                  trainable=True)
        else:
            self.input_variable = None

        # If there are weight variables, prepare a params tensor as a trainable variable
        if self.weight_vars:
            # Initialize the variable tensor that will hold the weights/parameters/angles
            initializer = tf.constant_initializer(np.random.uniform(low=0., high=2 * np.pi, size=len(self.weight_vars)))
            self.weight_variable = self.add_weight(name="params_tensor_variable",
                                                   shape=(len(self.weight_vars)),
                                                   dtype=self._cast_type,
                                                   initializer=initializer,
                                                   trainable=True)

            # If the user specified initial values for the parameters, use them
            if compile_args is not None and compile_args["initial_values"] is not None:
                # Assign them in the order given by self.second
                toVariable = [self.second[i] for i in self.second]  # Variable names in the correct order
                self.weight_variable.assign([compile_args["initial_values"][val]
                                             for val in toVariable])
        else:
            self.weight_variable = None

        # Store extra useful information
        self._input_len = 0
        if input_vars:
            self._input_len = len(self.input_vars)
        self._params_len = len(list(self.weight_vars))

        self.samples = None
        if self.compile_args is not None:
            self.samples = self.compile_args["samples"]

    def __call__(self, input_tensor: tf.Tensor = None) -> tf.Tensor:
        """
        Calls the Objective on a TF tensor object and returns the results.

        There are three cases which we could have:
            1) We have just input variables
            2) We have just parameter variables
            3) We have both input and parameter variables

        We must determine which situation we are in and execute the corresponding _do() function to also get the
        correct gradients.

        Returns
        -------
        tf.Tensor:
            a TF tensor, the result of calling the underlying objective on the input combined with the parameters.
        """
        # This is for the situation where various different inputs are being introduced
        if input_tensor is not None:
            self.set_input_values(input_tensor)

        # Case of both inputs and parameters
        if self.input_vars and self.weight_vars:
            return self._do(self.get_inputs_variable(), self.get_params_variable())

        # Case of just inputs
        elif self.input_vars:
            return self._do_just_input(self.get_inputs_variable())

        # Case of just parameters
        return self._do_just_params(self.get_params_variable())

    @tf.custom_gradient
    def _do_just_input(self, input_tensor_variable: tf.Variable) -> (tf.Tensor, Callable):
        """
        Forward pass with just the inputs.

        This in-between function is necessary in order to have the custom gradient work in Tensorflow. That is the
        reason for returning the grad() function as well.

        Parameters
        ----------
        input_tensor_variable
            the tf.Variable which holds the values of the input

        Returns
        -------
        result
            The result of the forwarding
        """
        if input_tensor_variable.shape != self._input_len:
            raise TequilaMLException(
                'Received input of len {} when Objective takes {} inputs.'.format(len(input_tensor_variable.numpy()),
                                                                                  self._input_len))
        input_tensor_variable = tf.stack(input_tensor_variable)

        def grad(upstream):
            # Get the gradient values
            input_gradient_values = self.get_grads_values(only="inputs")

            # Convert to tensor
            in_Tensor = tf.convert_to_tensor(input_gradient_values, dtype=self._cast_type)

            # Right-multiply the upstream
            in_Upstream = tf.dtypes.cast(upstream, self._cast_type) * in_Tensor

            # Transpose and reduce sum
            return tf.reduce_sum(tf.transpose(in_Upstream), axis=0)

        return self.realForward(inputs=input_tensor_variable, params=None), grad

    @tf.custom_gradient
    def _do_just_params(self, params_tensor_variable: tf.Variable) -> (tf.Tensor, Callable):
        """
        Forward pass with just the parameters

        This in-between function is necessary in order to have the custom gradient work in Tensorflow. That is the
        reason for returning the grad() function as well.

        Parameters
        ----------
        params_tensor_variable
            the tf.Variable which holds the values of the parameters

        Returns
        -------
        result
            The result of the forwarding
        """
        if params_tensor_variable.shape != self._params_len:
            raise TequilaMLException(
                'Received input of len {} when Objective takes {} inputs.'.format(len(params_tensor_variable.numpy()),
                                                                                  self._input_len))
        params_tensor_variable = tf.stack(params_tensor_variable)

        def grad(upstream):
            # Get the gradient values
            parameter_gradient_values = self.get_grads_values(only="params")

            # Convert to tensor
            par_Tensor = tf.convert_to_tensor(parameter_gradient_values, dtype=self._cast_type)

            # Right-multiply the upstream
            par_Upstream = tf.dtypes.cast(upstream, self._cast_type) * par_Tensor

            # Transpose and reduce sum
            return tf.reduce_sum(tf.transpose(par_Upstream), axis=0)

        return self.realForward(inputs=None, params=params_tensor_variable), grad

    @tf.custom_gradient
    def _do(self, input_tensor_variable: tf.Variable, params_tensor_variable: tf.Variable) -> (tf.Tensor, Callable):
        """
        Forward pass with both input and parameter variables

        This in-between function is necessary in order to have the custom gradient work in Tensorflow. That is the
        reason for returning the grad() function as well.

        Parameters
        ----------
        input_tensor_variable
            the tf.Variable which holds the values of the input
        params_tensor_variable
            the tf.Variable which holds the values of the parameters

        Returns
        -------
        result
            The result of the forwarding
        """
        if params_tensor_variable.shape != self._params_len:
            raise TequilaMLException(
                'Received input of len {} when Objective takes {} inputs.'.format(len(params_tensor_variable.numpy()),
                                                                                  self._input_len))
        params_tensor_variable = tf.stack(params_tensor_variable)

        if input_tensor_variable.shape != self._input_len:
            raise TequilaMLException(
                'Received input of len {} when Objective takes {} inputs.'.format(len(input_tensor_variable.numpy()),
                                                                                  self._input_len))
        input_tensor_variable = tf.stack(input_tensor_variable)

        def grad(upstream):
            input_gradient_values, parameter_gradient_values = self.get_grads_values()
            # Convert to tensor
            in_Tensor = tf.convert_to_tensor(input_gradient_values, dtype=self._cast_type)
            par_Tensor = tf.convert_to_tensor(parameter_gradient_values, dtype=self._cast_type)

            # Multiply with the upstream
            in_Upstream = tf.dtypes.cast(upstream, self._cast_type) * in_Tensor
            par_Upstream = tf.dtypes.cast(upstream, self._cast_type) * par_Tensor

            # Transpose and sum
            return tf.reduce_sum(tf.transpose(in_Upstream), axis=0), tf.reduce_sum(tf.transpose(par_Upstream), axis=0)

        return self.realForward(inputs=input_tensor_variable,
                                params=params_tensor_variable), grad

    def realForward(self, inputs: Union[tf.Variable, None], params: Union[tf.Variable, None]) -> tf.Tensor:
        """
        This is where we really execute the forward pass.

        Parameters
        ----------
        inputs
            tf.Variable of the inputs
        params
            tf.Variable of the parameters

        Returns
        -------
            The result of the forwarding
        """

        def tensor_fix(inputs_tensor: Union[tf.Tensor, None], params_tensor: Union[tf.Tensor, None],
                       first: Dict[int, Variable], second: Dict[int, Variable]):
            """
            Prepare a dict with the right information about the involved variables (whether input or parameter) and
            their corresponding values.

            Note: if "inputs_tensor" and "angles_tensor" are None or "first" and "second" are empty dicts, something
            went wrong, since the objective should have either inputs or parameters to tweak.

            Parameters
            ----------
            inputs_tensor
                Tensor holding the values of the inputs
            params_tensor
                Tensor holding the values of the parameters
            first
                Dict mapping numbers to input variable names
            second
                Dict mapping numbers to parameter variable names

            Returns
            -------
            variables
                Dict mapping all variable names to values
            """
            variables = {}
            if inputs_tensor is not None:
                for i, val in enumerate(inputs_tensor):
                    variables[first[i]] = val.numpy()
            if params_tensor is not None:
                for i, val in enumerate(params_tensor):
                    variables[second[i]] = val.numpy()
            return variables

        variables = tensor_fix(inputs, params, self.first, self.second)
        result = self.comped_objective(variables=variables, samples=self.samples)
        if not isinstance(result, np.ndarray):
            # this happens if the Objective is a scalar since that's usually more convenient for pure quantum stuff.
            result = np.array(result)
        if hasattr(inputs, 'device'):
            if inputs.device == 'cuda':
                return tf.convert_to_tensor(result).to(inputs.device)
            else:
                return tf.convert_to_tensor(result)
        return tf.convert_to_tensor(result)

    def get_grads_values(self, only: str = None):
        """
        Gets the values of the gradients with respect to the inputs and the parameters.

        You can specify whether you want just the input or parameter gradients for the sake of efficiency.

        Returns
        -------
        grad_values
            If "only" is None, a tuple of two elements, the first one being a list of gradients to apply to the input
            variables, and the second element being a list of gradients to apply to the parameter variables.
            If only == inputs, just the list of gradient values w.r.t. the input variables.
            If only == params, just the list of gradient values w.r.t. the parameter variables.
        """
        get_input_grads = True
        get_param_grads = True

        # Determine which gradients to calculate
        if only is not None:
            if only == "inputs":
                get_input_grads = True
                get_param_grads = False
            elif only == "params":
                get_input_grads = False
                get_param_grads = True
            else:
                raise TequilaMLException("Valid values for \"only\" are \"inputs\" and \"params\".")

        # Get the current values of the inputs and parameters in a dict called "variables"
        variables = {}

        # Inputs
        list_inputs = self.get_inputs_list()
        if list_inputs:
            for i in self.first:
                variables[self.first[i]] = list_inputs[i]

        # Parameters
        list_angles = self.get_params_list()
        if list_angles:
            for w in self.second:
                variables[self.second[w]] = list_angles[w]

        # GETTING THE GRADIENT VALUES
        # Get the gradient values with respect to the inputs
        inputs_grads_values = []
        if get_input_grads and self.first:
            for in_var in self.first.values():
                self.fill_grads_values(inputs_grads_values, in_var, variables, self.i_grads)

        # Get the gradient values with respect to the parameters
        param_grads_values = []
        if get_param_grads and self.second:
            for param_var in self.second.values():  # Iterate through the names of the parameters
                self.fill_grads_values(param_grads_values, param_var, variables, self.w_grads)

        # Determine what to return
        if get_input_grads and get_param_grads:
            return inputs_grads_values, param_grads_values
        elif get_input_grads and not get_param_grads:
            return inputs_grads_values
        elif not get_input_grads and get_param_grads:
            return param_grads_values

    def set_input_values(self, initial_input_values: Union[dict, tf.Tensor]):
        """
        Stores the values of the tensor into the self.input_variable. Intended to be used to set the values that the
        input variables initially will have before training.

        Parameters
        ----------

        """
        # If the input is a dictionary
        if isinstance(initial_input_values, dict):
            input_values_tensor = tf.convert_to_tensor([initial_input_values[i] for i in self.first.values()])

            # Check that input variables are expected
            if self.input_vars is not None:
                # Check that the length of the tensor of the variable is the correct one
                if input_values_tensor.shape == self._input_len:
                    self.input_variable.assign(input_values_tensor)
                else:
                    raise TequilaMLException("Input tensor has shape {} which does not match "
                                             "the {} inputs expected".format(input_values_tensor.shape,
                                                                             self._input_len))
            else:
                raise TequilaMLException("No input variables were expected.")
        # If the input is a tensor
        elif isinstance(initial_input_values, tf.Tensor):
            if initial_input_values.shape == self._input_len:
                # We have no information about which value corresponds to which variable, so we assume that the user
                # knows that the order will be the same as in self.first
                self.input_variable.assign(initial_input_values)
            else:
                raise TequilaMLException("Input tensor has shape {} which does not match "
                                         "the {} inputs expected".format(initial_input_values.shape, self._input_len))

    def fill_grads_values(self, grads_values, var, variables, objectives_grad):
        """
        Inserts into "grads_values" the gradient values per objective in objectives_grad[var], where var is the name
        of the variable.

        Parameters
        ----------
        grads_values
            List in which we insert the gradient values (No returns)
        var
            Variable over which we are calculating the gradient values
        variables
            Dict mapping all variables to their current values
        objectives_grad
            List of ExpectationValueImpls that will be simulated to calculate the gradient value of a given variable
        """
        var_results = []
        grads_wrt_var = objectives_grad[var]
        if not isinstance(grads_wrt_var, List):
            grads_wrt_var = [grads_wrt_var]
        for obj in grads_wrt_var:
            var_results.append(simulate(objective=obj, variables=variables,
                                        backend=self.compile_args["backend"],
                                        samples=self.samples))
        grads_values.append(var_results)

    def get_params_variable(self):
        return self.weight_variable

    def get_params_list(self):
        if self.get_params_variable() is not None:
            return self.get_params_variable().numpy().tolist()
        return []

    def get_inputs_variable(self):
        return self.input_variable

    def get_inputs_list(self):
        if self.get_inputs_variable() is not None:
            return self.get_inputs_variable().numpy().tolist()
        return []

    def get_input_values(self):
        # Tensor values is in the order of self.input_vars
        input_values = self.get_inputs_list()
        input_values_dict = {}
        for i, value in enumerate(self.input_vars):
            input_values_dict[value] = input_values[i]
        return input_values_dict

    def get_params_values(self):
        # Tensor values is in the order of self.weight_vars
        params_values = self.get_params_list()
        params_values_dict = {}
        for i, value in enumerate(self.weight_vars):
            params_values_dict[value] = params_values[i]
        return params_values_dict

    def set_cast_type(self, datatype):
        """
        The default datatype of this TFLayer is float32, since this is the most precise float supported by TF
        optimizers at the time of writing.

        This method is intended so that in the future, whenever TF optimizers support float64, the datatype cast to can
        be changed to float64. However, if for some reason you'd like to cast it to something else, you may, although it
        only really makes sense to cast it to float types since these are the values that the variables will have.

        Parameters
        ----------
        datatype
            Datatype to cast to. Expecting typing.Union[tf.float64, tf.float32, tf.float16].
        """
        self._cast_type = datatype

    def __repr__(self) -> str:
        string = 'Tequila TFLayer. Represents: \n'
        string += '{} \n'.format(str(self.objective))
        string += 'Current Weights: {}'.format(list(self.weight_vars))
        return string
