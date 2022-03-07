import numpy, typing
from .objective import Objective, ExpectationValueImpl, format_variable_dictionary
from tequila import TequilaException

class QTensor(numpy.ndarray):
    # see here: https://numpy.org/devdocs/user/basics.subclassing.html

    def __new__(subtype, objective_list=None, *args, **kwargs):
        if "dtype" not in kwargs:
            kwargs["dtype"] = Objective
        return super().__new__(subtype, *args, **kwargs)

    def __init__(self, objective_list=None, *args, **kwargs):
        super().__init__()
        # do all-zero initialization
        shape = self.shape
        if objective_list is None:
            with numpy.nditer(self, flags =["refs_ok"], op_flags=['readwrite']) as it:
                for x in it:
                    x[...] = Objective()
        else:
            j=0
            with numpy.nditer(self, flags =["refs_ok"], op_flags=['readwrite']) as it:
                for x in it:
                    x[...] = objective_list[j]
                    j=j+1

    def __call__(self,variables=None, *args, **kwargs):
        """
        Return the output of the calculation the objective represents.

        Parameters
        ----------
        variables: dict:
            dictionary instantiating all variables that may appear within the objective.
        args
        kwargs

        Returns
        -------
        float:
            the result of the calculation represented by this objective.
        """
        variables = format_variable_dictionary(variables)
        # failsafe
        check_variables = {k: k in variables for k in self.extract_variables()}
        if not all(list(check_variables.values())):
            raise TequilaException("Objective did not receive all variables:\n"
                                   "You gave\n"
                                   " {}\n"
                                   " but the objective depends on\n"
                                   " {}\n"
                                   " missing values for\n"
                                   " {}".format(variables, self.extract_variables(), [k for k,v in check_variables.items() if not v]))

        # avoid multiple evaluations
        evaluated = {}
        ev_array = []
        newtensor = self.flatten()
        for obj in newtensor:
            a = obj(variables=variables, *args, **kwargs)
            ev_array.append(a)
        ev_array = numpy.reshape(ev_array,self.shape)
        if ev_array.shape == ():
            return float(ev_array)
        elif len(ev_array) == 1:
            return float(ev_array[0])
        else:
            return ev_array

    def apply(self, fn):
        _f = self.HelperObject(func=fn)
        _fn = numpy.vectorize(_f)
        return _fn(self)

    def extract_variables(self)->list:
        newtensor = self.flatten()
        unique = []
        for obj in newtensor:
            if hasattr(obj, 'extract_variables'):
                var_list = obj.extract_variables()
                for j in var_list:
                    if j not in unique:
                        unique.append(j)
        return unique

    def get_expectationvalues(self):
        """
        Returns
        -------
        list:
            all the expectation values that make up the objective.
        """
        newtensor = self.flatten()
        expvals = []
        for obj in newtensor:
            if hasattr(obj, 'get_expectationvalues'):
                expvals += obj.get_expectationvalues()
        return expvals

    def count_measurements(self):
        """
        Count all measurements necessary for this objective:
        Function will iterate to all unique expectation values and count the
        number of Pauli strings in the corresponding Hamiltonians
        Returns
        -------
        Number of measurements required for this objective
        Measurements can be on different circuits (with regards to gates, depth, size, qubits)
        """
        return sum(E.count_measurements() for E in list(set(self.get_expectationvalues())))

    def count_expectationvalues(self, unique=True):
        """
        Parameters
        ----------
        unique: bool:
            whether or not to count identical expectationvalues as distinct.

        Returns
        -------
        int:
            how many (possibly, how many unique) expectationvalues are contained within the objective.

        """
        if unique:
            return len(set(self.get_expectationvalues()))
        else:
            return len(self.get_expectationvalues())

    def __repr__(self):
        _repmat = numpy.empty(self.shape,dtype = object)
        _repmat = _repmat.flatten()
        newtensor = self.flatten()
        for i in range(len(newtensor)):
            _repmat[i] = repr(newtensor[i])
        _repmat = _repmat.reshape(self.shape)
        return repr(_repmat)

    def __str__(self):
        variables = self.extract_variables()
        if len(variables) > 5:
            variables = len(variables)
        newtensor = self.flatten()
        types = []
        for obj in newtensor:
            if hasattr(obj, 'get_expectationvalues'):
                _types = [type(E) for E in obj.get_expectationvalues()]
                for tt in _types:
                    types.append(tt)
        types = list(set(types))
        if ExpectationValueImpl in types:
            if len(types) == 1:
                types = "not compiled"
            else:
                types = "partially compiled to " + str([t for t in types if t is not ExpectationValueImpl])

        unique = self.count_expectationvalues(unique=True)
        measurements = self.count_measurements()
        return "QTensor of shape {} with {} unique expectation values\n" \
              "total measurements = {}\n" \
              "variables          = {}\n" \
              "types              = {}".format(self.shape, unique, measurements, variables, types)

    def contract(self):
        newtensor = self.flatten()
        out_array=[obj for obj in newtensor]
        summed = out_array[0]
        for entry in out_array[1:]:
            summed += entry
        return summed


    class HelperObject:
        """
        This is a small helper object class for tequila objectives
        it is within the QTensor class to shield it from the outside (meant for internal use)
        create if like this:
            ff = HelperObject(func=f) where f is the function you want to apply later (e.g. numpy.sin)
        use if like this with tequila objectives
            f_on_objective = ff(objective) 
        """
        def __init__(self, func):
            self.func = func
        def __call__(self, objective):
            return objective.apply(self.func)

# ------------------------------------------------------
# backward compatibility with old VectorObjective class
# ------------------------------------------------------

def vectorize(objectives):
    """
    Combine several objectives in order, into one longer vector.

    Parameters
    ----------
    objectives: iterable:
        the objectives to combine as a vector. Note that this is not addition, but the 'end to end' combination of
        vectors; the new objective will have length Sum(len(x) for x in objectives)

    Returns
    -------
    QTensor:
        Objectives stacked together.
    """
    return QTensor(objective_list=objectives, shape=(len(objectives),))

def VectorObjective(argsets: typing.Iterable = None, transformations: typing.Iterable[callable] = None):
    if argsets is None:
        return QTensor()
    objective_list = []
    if transformations is None:
        for argset in argsets:
            objective_list.append(Objective(args=argset))
    else:
        assert len(argsets) == len(transformations)
        for i in range(len(argsets)):
            objective_list.append(Objective(args=argsets[i], transformation=transformations[i]))

    return vectorize(objectives=objective_list)

