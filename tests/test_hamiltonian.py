import unittest
# from .context import openvqe
from openvqe.hamiltonian.hamiltonian_qc import HamiltonianQC
from openvqe.parameters import ParametersQC, ParametersPsi4
import openfermion

class TestParameters(unittest.TestCase):

    def test_qc(self):

        geomstrings = [
            " H 0.0 0.0 1.0\n H 0.0 0.0 -1.0",
            " he 0.0 0.0 0.0",
            " Li 0.0 0.0 0.377\n h 0.0 0 -1.13"
        ]

        bases = [
            'sto-3g',
            '6-31g'
        ]

        for geom in geomstrings:
            for basis in bases:
                if basis!='sto-3g' and 'Li' in geom: continue
                parameters_qc = ParametersQC(geometry=geom, basis_set=basis)
                hqc = HamiltonianQC(parameters_qc)
                Hmol=hqc.get_molecular_Hamiltonian()
                H=hqc.get_Hamiltonian()

                self.assertEqual(H, openfermion.jordan_wigner(openfermion.get_fermion_operator(Hmol)))

