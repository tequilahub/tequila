import tequila as tq
import numpy as np
import time
import os
import csv
from tequila.hamiltonian import PauliString
import pytest


#!/usr/bin/env python
import tequila as tq
import numpy as np
import time

print("Test: Circuit und Hamiltonian mit Variablen (SPEX vs. Qulacs)")

# --- Parameterdefinition ---
# Definiere mehrere Variablen für die Rotation
a = tq.Variable("a")
b = tq.Variable("b")
c = tq.Variable("c")
variables = {"a": np.pi/3, "b": np.pi/4, "c": np.pi/6}

# --- Circuitaufbau ---
# Erzeuge einen Circuit, der auf 3 Qubits operiert:
# - Eine Rx-Rotation auf Qubit 0 (Winkel "a")
# - Eine Ry-Rotation auf Qubit 1 (Winkel "b")
# - Eine Rz-Rotation auf Qubit 2 (Winkel "c")
# - Zusätzlich eine parametrische exponentielle Pauli-Rotation (ExpPauli) auf Qubit 0 und 2 (Pauli-String "X(0)Z(2)")
U = tq.gates.Rx(angle="a", target=(0,)) \
    + tq.gates.Ry(angle="b", target=(1,)) \
    + tq.gates.Rz(angle="c", target=(2,))
U += tq.gates.ExpPauli(angle="a", paulistring="X(0)Z(2)")

print("\nCircuit U:")
print(U)

# --- Hamiltonianaufbau ---
# Erstelle einen zusammengesetzten Hamiltonian auf 3 Qubits,
# z.B.: H = Z(0) + X(1)Y(2) + Z(0)Z(1)
H = tq.QubitHamiltonian("Z(0) + X(1)Y(2) + Z(0)Z(1)")
print("\nHamiltonian H:")
print(H)

# Erzeuge ein Erwartungswertobjekt
E = tq.ExpectationValue(U=U, H=H)

# --- Simulation mit SPEX ---
start = time.time()
wfn_spex = tq.simulate(U, variables, backend='spex')
exp_spex = tq.simulate(E, variables, backend='spex')
end = time.time()
time_spex = end - start

# --- Simulation mit Qulacs ---
start = time.time()
wfn_qulacs = tq.simulate(U, variables, backend='qulacs')
exp_qulacs = tq.simulate(E, variables, backend='qulacs')
end = time.time()
time_qulacs = end - start

# --- Ergebnisse ausgeben ---
print("\nSimulationsergebnisse:")
print("Wellenfunktion (SPEX backend):")
print(wfn_spex)
print("\nWellenfunktion (Qulacs backend):")
print(wfn_qulacs)

print("\nErwartungswert (SPEX backend):", exp_spex, f"(Simulationszeit: {time_spex:.3f}s)")
print("Erwartungswert (Qulacs backend):", exp_qulacs, f"(Simulationszeit: {time_qulacs:.3f}s)")

# Optional: Vergleiche das innere Produkt der beiden Wavefunctions (Quadrat des Betrags)
inner_prod = np.abs(wfn_spex.inner(wfn_qulacs))**2
print("\nInneres Produkt (Quadrat) zwischen SPEX und Qulacs:", inner_prod)



"""
print("\nTest: 1")
U = tq.gates.ExpPauli(paulistring=PauliString({0: "X", 1: "Z"}), angle=np.pi / 2)
H = tq.QubitHamiltonian("Z(0)")
E = tq.ExpectationValue(U=U, H=H)

print(U)
print("spex-U:", tq.simulate(U, backend='spex'))
print("qulacs-U:", tq.simulate(U, backend='qulacs'))
print("spex:", tq.simulate(E, backend='spex'))
print("qulacs:", tq.simulate(E, backend='qulacs'))


print("\nTest: 2")
H = tq.QubitHamiltonian("Z(0)X(1)")

U = tq.gates.Rx(angle=np.pi / 4, target=(0,))
U += tq.gates.Ry(angle=np.pi / 3, target=(1,)) 
E = tq.ExpectationValue(U=U, H=H)

print(U)
print("spex-U:", tq.simulate(U, backend='spex'))
print("qulacs-U:", tq.simulate(U, backend='qulacs'))
print("spex:", tq.simulate(E, backend='spex'))
print("qulacs:", tq.simulate(E, backend='qulacs'))


print("\nTest: 3")
H = tq.QubitHamiltonian("Z(0)")

U = tq.gates.X(target=(0,))
U += tq.gates.Y(target=(1,)) 
E = tq.ExpectationValue(U=U, H=H)

print(U)
print("spex-U:", tq.simulate(U, backend='spex'))
print("qulacs-U:", tq.simulate(U, backend='qulacs'))
print("spex:", tq.simulate(E, backend='spex'))
print("qulacs:", tq.simulate(E, backend='qulacs'))


print("\nTest: 4")
U = (
    tq.gates.Rx(angle=np.pi / 2, target=(0,)) +
    tq.gates.Rz(angle=np.pi / 2, target=(1,)) +
    tq.gates.Ry(angle=np.pi / 3, target=(2,)) +
    tq.gates.ExpPauli(paulistring=PauliString({0: "X", 1: "X", 2: "X"}), angle=np.pi / 4)
)

H = tq.QubitHamiltonian("X(0)X(1)X(2) + Z(0) + Z(1) + Z(2)")
E = tq.ExpectationValue(U=U, H=H)

print(U)
print("spex-U:", tq.simulate(U, backend='spex'))
print("qulacs-U:", tq.simulate(U, backend='qulacs'))
print("spex:", tq.simulate(E, backend='spex'))
print("qulacs:", tq.simulate(E, backend='qulacs'))

print("\nTest: 5")
os.environ["OMP_NUM_THREADS"] = "6"

results = []

for n in range(1, 15):
    # 1) Geometrie aufbauen
    geom = ""
    for k in range(2*n):
        geom += f"h 0.0 0.0 {1.5*k}\n"

    # 2) Molekül + Hamiltonian
    mol = tq.Molecule(geometry=geom, basis_set="sto-3g")
    H   = mol.make_hardcore_boson_hamiltonian()

    # 3) Ansatz U
    edges = [(2*i, 2*i+1) for i in range(n)]
    U = mol.make_ansatz(name="HCB-SPA", edges=edges)
    # Alle im Ansatz auftretenden Variablen auf 1.0 setzen
    U = U.map_variables({var: 1.0 for var in U.extract_variables()})

    # 4) Erwartungswert-Objekt
    E_obj = tq.ExpectationValue(H=H, U=U)

    # -- SPEX-Berechnung --
    start_spex = time.time()
    E_spex = tq.simulate(E_obj, backend='spex', num_threads=6)
    end_spex = time.time()
    time_spex = end_spex - start_spex

    # -- Qulacs-Berechnung --
    if n <= 10:
        start_qulacs = time.time()
        E_qulacs = tq.simulate(E_obj, backend='qulacs')
        end_qulacs = time.time()
        time_qulacs = end_qulacs - start_qulacs

    total_measurements = E_obj.count_measurements()

    # Speichern der Daten
    results.append({
        'n': n,
        'total_measurements' : total_measurements,
        'E_spex': E_spex,
        'time_spex': time_spex,
        'E_qulacs': E_qulacs,
        'time_qulacs': time_qulacs
    })

    if E_qulacs is not None:
        print(f"n={n:2d} | total_measurements={total_measurements} | "
              f"E_spex={E_spex:.6f} (dt={time_spex:.2f}s) | "
              f"E_qulacs={E_qulacs:.6f} (dt={time_qulacs:.2f}s)")
    else:
        print(f"n={n:2d} | total_measurements={total_measurements} | "
              f"E_spex={E_spex:.6f} (dt={time_spex:.2f}s) | "
              f"E_qulacs=--- (dt=---)  (für n>13 nicht berechnet)")
    
    with open("spex_qulacs_comparison.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        # Kopfzeile
        writer.writerow(["n", "total_measurements", "E_spex", "time_spex", "E_qulacs", "time_qulacs"])
        # Datenzeilen
        for entry in results:
            writer.writerow([
                entry["n"],
                entry["total_measurements"],
                entry["E_spex"],
                entry["time_spex"],
                entry["E_qulacs"] if entry["E_qulacs"] is not None else "NA",
                entry["time_qulacs"] if entry["time_qulacs"] is not None else "NA"
            ])
    
    E_qulacs = None

"""