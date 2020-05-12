from tequila.hamiltonian import PauliString

if __name__ == "__main__":
    testdata = {0: 'x', 1: 'y', 2: 'z'}
    test = PauliString(data=testdata, coeff=2)
    print("test=", test)
    print("test: openfermion_key = ", test.key_openfermion())
    print("reinitialized: ", PauliString.from_openfermion(key=test.key_openfermion(), coeff=test.coeff))
