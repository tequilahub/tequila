import tequila as tq
import pytest
from math import pi, sqrt
import numpy as np
import random

# Fixed angles
# formatted as key = angle as string ; value = actual angle 
def get_fixed_angles():
    return {
        "0": 0,
        "π/2": 0.5 * pi,
        "π": pi,
        "3π/2": 1.5 * pi,
        "2π": 2 * pi,
        "1.2": 1.2,
        "0.23": 0.23,
        "-4.123": -4.123,
        "π/6": pi / 6,
        "-π/4": -pi / 4,
        "2.718": 2.718,
        "-0.99": -0.99,
        "√2": sqrt(2),
        "5π/3": 5 * pi / 3,
        "7.77": 7.77,
        "-3π": -3 * pi,
    }


# a dictionary with random angles generated from the same seed 
def get_random_angles():
    random.seed(42)
    angles = {}
    for _ in range(16):
        angle_val = random.uniform(-4 * pi, 4 * pi)
        label = f"{angle_val:.3f}"
        angles[label] = angle_val
    return angles


@pytest.mark.parametrize("angles", [get_fixed_angles(), get_random_angles()])
def test_only_hadamard_and_global_phase(angles):
    for angle_name, angle_val in angles.items():
        # compute expected phase from the angle information 
        expected_phase = np.exp(1j * angle_val)

        U = tq.QCircuit()
        U.n_qubits = 1
        U += tq.gates.H(0)
        simulation_without_phase = tq.simulate(U)

        U += tq.gates.GlobalPhase(angle=angle_val)
        simulation_with_gp = tq.simulate(U)

        simulation_without_phase *= expected_phase
        assert simulation_with_gp.isclose(simulation_without_phase, ignore_global_phase=False), f"Failed at angle {angle_name}"


@pytest.mark.parametrize("angles", [get_fixed_angles(), get_random_angles()])
def test_bell_circuit_and_global_phase(angles):
    for angle_name, angle_val in angles.items():
        # compute expected phase from the angle information 
        expected_phase = np.exp(1j * angle_val)
        U = tq.QCircuit()
        U.n_qubits = 2
        U += tq.gates.H(0)
        U += tq.gates.CNOT(target=1, control=0)
        simulation_without_phase = tq.simulate(U)

        U += tq.gates.GlobalPhase(angle=angle_val)
        simulation_with_gp = tq.simulate(U)

        simulation_without_phase *= expected_phase
        assert simulation_with_gp.isclose(simulation_without_phase, ignore_global_phase=False), f"Failed at angle {angle_name}"


@pytest.mark.parametrize("angles", [get_fixed_angles(), get_random_angles()])
def test_mixed_gates_with_global_phase_in_between(angles):
    for angle_name, angle_val in angles.items():
        expected_global_phase_factor = np.exp(1j * angle_val)

        # Circuit with global phases
        circuit_with_phase = tq.QCircuit()
        circuit_with_phase.n_qubits = 3
        circuit_with_phase += tq.gates.GlobalPhase(angle=angle_val)
        circuit_with_phase += tq.gates.H(0)
        circuit_with_phase += tq.gates.GlobalPhase(angle=angle_val)
        circuit_with_phase += tq.gates.CNOT(target=1, control=0)
        circuit_with_phase += tq.gates.CNOT(target=2, control=1)
        circuit_with_phase += tq.gates.H(2)
        circuit_with_phase += tq.gates.GlobalPhase(angle=angle_val)
        circuit_with_phase += tq.gates.GlobalPhase(angle=angle_val)

        # Circuit without global phases
        circuit_without_phase = tq.QCircuit()
        circuit_without_phase.n_qubits = 3
        circuit_without_phase += tq.gates.H(0)
        circuit_without_phase += tq.gates.CNOT(target=1, control=0)
        circuit_without_phase += tq.gates.CNOT(target=2, control=1)
        circuit_without_phase += tq.gates.H(2)

        # Simulations
        wf_with_phase = tq.simulate(circuit_with_phase)
        wf_without_phase = tq.simulate(circuit_without_phase)

        # Multiply by total phase factor (4 global phases)
        wf_without_phase *= expected_global_phase_factor ** 4

        # Assertion
        assert wf_with_phase.isclose(wf_without_phase, ignore_global_phase=False), f"Failed at angle {angle_name}"
