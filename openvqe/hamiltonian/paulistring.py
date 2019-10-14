"""
Convenient DataClass for single PauliStrings
Internal Storage is a dictionary where keys are particle-numbers and values the primitive paulis
i.e. X(1)Y(2)Z(5) is {1:'x', 2:'y', 5:'z'}
additional a coefficient can be stored
iteration is then over the dimension
"""

from openvqe.tools.convenience import number_to_string

class PauliString:

    def key_openfermion(self):
        """
        Convert into key to store in Hamiltonian
        Same key syntax than openfermion
        :return: The key for the openfermion dataformat
        """
        key = []
        for k, v in self._data.items():
            key.append((k, v))
        return tuple(key)

    def __repr__(self):
        result = number_to_string(self.coeff)
        for k,v in self._data.items():
            result += str(v)+"("+str(k)+")"
        return  result

    def __init__(self, data=None, coeff=None):
        if data is None:
            self._data = {}
        else:
            # stores the paulistring as dictionary
            # keys are the dimensions
            # values are x,y,z
            self._data = data
        self._coeff = coeff

    def items(self):
        return self._data.items()

    @classmethod
    def init_from_openfermion(cls, key, coeff=None):
        data = {}
        for term in key:
            index = term[0]
            pauli = term[1]
            data[index] = pauli
        return PauliString(data=data, coeff=coeff)


    @property
    def coeff(self):
        if self._coeff is None:
            return 1
        else:
            return self._coeff

    @coeff.setter
    def coeff(self, other):
        self._coeff = other
        return self

    def __eq__(self, other):
        return self._data == other._data


if __name__ == "__main__":

    testdata = {0:'x', 1:'y', 2:'z'}
    test = PauliString(data=testdata, coeff=2)
    print("test=", test)
    print("test: openfermion_key = ", test.key_openfermion())
    print("reinitialized: ", PauliString.init_from_openfermion(key=test.key_openfermion(), coeff=test.coeff))
