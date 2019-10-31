from openvqe.circuit import gates
from openvqe.circuit import Variable
import sympy

if __name__ == "__main__":
    """
    Simple String Based initialization
    """
    U = gates.Rx(target=0, angle="param0") + gates.Ry(target=1, control=0, angle="param1") + gates.Rz(target=0, angle="param0")

    print(U)

    parameters = U.extract_parameters()

    print(parameters)

    parameters["param0"] = 2.0
    parameters["param1"] = 4.0

    U.update_parameters(parameters=parameters)

    print(U)

    """
    Initialization with Variables
    """

    param0 = Variable(name="param0", value=1.0)
    param1 = Variable(name="param1", value=2.0)

    U = gates.Rx(target=0, angle=param0) + gates.Ry(target=1, control=0, angle=param1) + gates.Rz(target=0, angle=-param0/2)

    print(U)

    parameters = U.extract_parameters()

    print(parameters)

    parameters["param0"] = 2.0
    parameters["param1"] = 4.0

    U.update_parameters(parameters=parameters)

    print(U)


