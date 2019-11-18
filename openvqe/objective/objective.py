import numpy, typing, copy

"""
Preliminary structure to carry information over to backends
Needs to be restructured and clarified but currently does the job
"""


# todo A lot!

class Objective:

    @property
    def unitaries(self):
        if self._unitaries is None:
            return []
        else:
            return self._unitaries

    @unitaries.setter
    def unitaries(self, u):
        if u is None:
            self._unitaries = u
        elif hasattr(u, "__iter__") or hasattr(u, "__get_item__"):
            self._unitaries = u
        else:
            self._unitaries = [u]

    def extract_parameters(self) -> typing.Dict[str, float]:
        """
        :return: All parameters of the objective
        """
        parameters = dict()
        for U in self.unitaries:
            parameters = {**parameters, **U.extract_parameters()}
        return parameters

    def update_parameters(self, parameters: typing.Dict[str, float]):
        """
        Update parameters of all unitaries
        :param parameters: parameters to update
        :return: self for chaining
        """
        for U in self.unitaries:
            U.update_parameters(parameters=parameters)
        return self

    def to_backend(self, simulator):
        out = Objective(unitaries=[], observable=[])
        for U in self.unitaries:
            out.unitaries += [simulator.backend_handler.recompile(abstract_circuit=U.decompose())]
            out.unitaries[-1].weight = U.weight
        out.observable = self.observable
        return out

    def __init__(self, observable=None, unitaries=None):
        self.unitaries = copy.deepcopy(unitaries)
        self.observable = copy.deepcopy(observable)

    def __eq__(self, other):

        if len(self.unitaries) != len(other.unitaries):
            return False

        for i, U in enumerate(self.unitaries):
            if U != other.unitaries[i]:
                print("oha \n", U, "\n", other.unitaries[i])
                return False

        return True

    def __add__(self, other):
        assert (self.observable == other.observable)
        return Objective(unitaries=self.unitaries + other.unitaries, observable=self.observable)

    def __mul__(self, other):
        # todo comming soon
        raise NotImplementedError("* not implemented yet")

    def objective_function(self, values, weights=None):
        """
        The abstract function which defines the operation performed on the expectation values
        The default is summation
        Overwrite this function to get different functions

        Potentially better Idea for the future: Maybe just use objectives as primitives, since they will have +,-,*,...,
        So the functions can be created and differentiated from the outside
        Then overwriting this functions is not necessary anymore

        :param values: Measurement results corresponding to <Psi_i|H|Psi_i> with |Psi_i> = U_i|Psi>
        :param weights: weights on the measurements
        :return:
        """
        if weights is None:
            weights = numpy.asarray([1] * len(values))
        else:
            weights = numpy.asarray(weights)
        values = numpy.asarray(values)
        assert (len(weights) == len(values))
        return weights.dot(values)

    def __repr__(self):
        nu = 0
        if self.unitaries is not None:
            nu = len(self.unitaries)
        no = "no "
        if self.observable is not None:
            no = "with "
        return "Objective(" + str(nu) + " unitaries, " + str(no) + " observable"

    def __str__(self):

        nu = "no"
        if self.unitaries is not None:
            nu = str(len(self.unitaries))

        result = "Objective " + nu + " Unitaries:\n"
        for U in self.unitaries:
            result += str(U) + "\n"

        if self.observable is None:
            result += "No Observable\n"
        else:
            result += "Observable:\n"
            result += str(self.observable)

        return result
