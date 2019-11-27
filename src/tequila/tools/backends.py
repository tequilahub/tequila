def initialize_simulator(simulator_type:str):
    """
    And it will allow for eay switching of backens to play around
    :param simulator_type: 'cirq', 'pyquil', 'symbolic' (default), 'qiskit'
    :return: the initialized simulators backend
    """
    # moving import statements to here, so the example also runs when not all are installed
    from tequila.simulators.simulator_cirq import SimulatorCirq
    from tequila.simulators.simulator_pyquil import SimulatorPyquil
    from tequila.simulators.simulator_symbolic import SimulatorSymbolic
    from tequila.simulators.simulator_qiskit import SimulatorQiskit

    if simulator_type.lower() == "cirq":
        return SimulatorCirq()
    elif simulator_type.lower() == 'pyquil':
        return SimulatorPyquil()
    elif simulator_type.lower() == 'qiskit':
        return SimulatorQiskit()
    else:
        return SimulatorSymbolic()