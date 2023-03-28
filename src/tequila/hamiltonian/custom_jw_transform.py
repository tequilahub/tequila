import typing
from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian
import tequila.hamiltonian.paulis as paulis
import openfermion


def _custom_transform(fermion: str, qubits: list) -> QubitHamiltonian:
    """
    This method maps fermion to qubits.

    Precondition:
    The last index of qubits should be greater than
    the site value of fermion.
    For instance fermion: "3", qubits: [1, 2, 3, 4] is allowed, but
    fermion: "3", qubits: [1, 2, 3] is not allowed

    Post condition:
    QubitHamiltonian of the qubits representation is returned

    === Parameter ===
    fermion: string representation of fermion operator to be mapped
    qubits: list of qubits on which Paulis are applied
    e.g. cjw.custom_transform("3^", [0, 1, 3, 5])

    === Return ===
    QubitHamiltonian
    """
    if not _is_input_compatible(fermion, qubits):
        raise Exception("Incorrect Input Format")

    qubit_hamiltonian = QubitHamiltonian()
    operator = fermion[0]
    site = qubits[int(operator[0])]
    x_op = 'X(' + str(site) + ')'
    pauli_x = QubitHamiltonian(x_op)
    y_op = 'Y(' + str(site) + ')'
    pauli_y = QubitHamiltonian(y_op)
    if fermion[-1] == '^':
        pauli_z = QubitHamiltonian("1")
        for qubit in range(0, int(operator)):
            z_op = 'Z(' + str(qubits[qubit]) + ')'
            pauli_z *= QubitHamiltonian(z_op)

        qubit_hamiltonian += pauli_z * pauli_x * 0.5

        qubit_hamiltonian += pauli_z * pauli_y * -0.5j

    else:
        pauli_z = QubitHamiltonian("1")
        for qubit in range(0, int(operator)):
            z_op = 'Z(' + str(qubits[qubit]) + ')'
            pauli_z *= QubitHamiltonian(z_op)

        qubit_hamiltonian += pauli_z * pauli_x * 0.5
        qubit_hamiltonian += pauli_z * pauli_y * 0.5j

    return qubit_hamiltonian


def _custom_transform_one_fermion(
        fermion: str, qubits: list,
        qubit_map: typing.Optional[dict] = None) -> QubitHamiltonian:
    """
    This method maps one fermion operator to qubit with specification of
    the mapping allowed

    Precondition:
    The last index of qubits should be greater than
    the site value of fermion. (Example in custom_transform method)
    The qubit specified by qubit_map should be contained in qubits and
    should not be duplicated in qubits.

    Post condition:
    Return QubitHamiltonian based on the custom transformation the user
    specified with qubits and qubit_map

    === Parameter ===
    fermion: str e.g. '3' '4^'
    qubits: list[int]
    qubit_map: dict[str, int], default=None

    === Return ===
    QubitHamiltonian

    """

    qubit_hamiltonian = QubitHamiltonian()
    revised_qubits = qubits
    mapped_qubits = False
    if qubit_map is None:
        qubit_hamiltonian += _custom_transform(fermion, qubits)
        return qubit_hamiltonian
    for key in qubit_map:
        if key == fermion and qubits.count(qubit_map[key]) > 1:
            raise Exception("Duplicated qubits. Either change "
                            "the mapped qubit or qubits list")
        if key == fermion and qubit_map[key] not in qubits:
            raise Exception("The mapped qubit should be contained in "
                            "qubits list")
        if key == fermion:
            revised_qubits.remove(qubit_map[key])
            revised_qubits.insert(int(key[0]), qubit_map[key])
            qubit_hamiltonian += _custom_transform(
                fermion, revised_qubits)
            mapped_qubits = True
    if not mapped_qubits:
        qubit_hamiltonian += _custom_transform(fermion, qubits)
    return qubit_hamiltonian


def custom_jw_transform(
        fermions: typing.Union[str, list], qubits: typing.Optional[list] = None,
        qubit_map: typing.Optional[dict] = None) -> QubitHamiltonian:
    """
    This method maps multiple fermion operators to specified qubits
    Precondition:
    Same as custom_transform_one_fermion

    Post condition:
    Return QubitHamiltonian based on the custom transformation the user
    specified with qubits and qubit_map

    === Parameter ===
    fermion: str e.g. '3' '3 4^ 6'
    qubits: list[list[int]] e.g. [[1, 2, 4, 5], [2, 3, 4, 6], [3, 5, 2, 3]]
    qubit_map: dict[str, int], default=None e.g. {'3': 4, '5^': 2}

    === Return ===
    QubitHamiltonian
    """

    if qubits is None:
        if qubit_map is None:
            largest_index = _find_largest_index(fermions)
            new_qubit_list = [i for i in range(largest_index + 1)]
            if _num_of_fermions(fermions) == 1:
                return custom_jw_transform(fermions, new_qubit_list, qubit_map)
            else:
                qubits_list = \
                    [new_qubit_list for _ in range(_num_of_fermions(fermions))]
                return custom_jw_transform(fermions, qubits_list, qubit_map)

    if not _is_input_compatible(fermions, qubits):
        raise Exception("Incorrect input given (fermions or qubits)."
                        " Follow the convention for fermions and qubits input")

    if isinstance(fermions, str):
        if len(fermions) <= 2:
            return _custom_transform_one_fermion(fermions, qubits,  qubit_map)
        fermion_ops = _split_fermions(fermions)
        qubit_hamiltonian = _custom_transform_one_fermion(
            fermion_ops[0], qubits[0], qubit_map)
        for i in range(1, len(fermion_ops)):
            qubit_hamiltonian *= _custom_transform_one_fermion(
                fermion_ops[i], qubits[i], qubit_map)

        return qubit_hamiltonian
    if isinstance(fermions, list):
        qubit_hamiltonian = _custom_transform_one_fermion(
            fermions[0], qubits[0], qubit_map)
        for i in range(1, len(fermions)):
            qubit_hamiltonian *= _custom_transform_one_fermion(
                fermions[i], qubits[i], qubit_map)

        return qubit_hamiltonian


def _num_of_fermions(fermions: typing.Union[str, list]) -> int:
    """

    """
    if isinstance(fermions, str):
        if len(fermions) <= 2:
            return 1
        return len(_split_fermions(fermions))
    elif isinstance(fermions, list):
        return len(fermions)


def _find_largest_index(fermions: typing.Union[str, list]) -> int:
    if isinstance(fermions, str):
        if len(fermions) <= 2:
            return int(fermions[0])
        fermion_ops = _split_fermions(fermions)
        max_ind = 0
        for index in fermion_ops:
            max_ind = max(max_ind, int(index[0]))
        return max_ind
    elif isinstance(fermions, list):
        max_ind = 0
        for index in fermions:
            max_ind = max(max_ind, int(index[0]))
        return max_ind


def _split_fermions(fermions: str) -> list:
    if ", " in fermions:
        fermion_ops = fermions.split(", ")
    elif " " in fermions:
        fermion_ops = fermions.split(" ")
    elif "," in fermions:
        fermion_ops = fermions.split(",")
    elif ":" in fermions:
        fermion_ops = fermions.split(":")
    else:
        raise Exception("Incorrect input format")
    return fermion_ops


def _is_input_compatible(
        fermion: typing.Union[str, list], qubits: list) -> bool:
    """
    Helper function for CustomJordanWigner class
    Checks if the inputs, namely the fermion operators are compatible with
    qubits in the following ways; the size, ...
    """
    if isinstance(fermion, str) and len(fermion) < 3:
        if int(fermion[0]) >= len(qubits):
            return False
        if len(qubits) != len(set(qubits)):
            return False
        return True
    elif isinstance(fermion, str) and len(fermion) >= 3:
        for maps in qubits:
            if not isinstance(maps, list):
                return False
        fermion_ops = _split_fermions(fermion)
        if len(fermion_ops) > len(qubits):
            return False
        for i in range(len(fermion_ops)):
            if not _is_input_compatible(fermion_ops, qubits[i]):
                return False
        return True
    return True
