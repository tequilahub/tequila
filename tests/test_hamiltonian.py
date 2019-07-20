import unittest
# from .context import openvqe
from openvqe.hamiltonian.hamiltonian_qc import HamiltonianQC, ParametersQC
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

        trafos = [
            'JW',
            'BK'
        ]

        for geom in geomstrings:
            for basis in bases:
                for trafo in trafos:
                    if basis!='sto-3g' and 'Li' in geom: continue
                    parameters_qc = ParametersQC(geometry=geom, basis_set=basis, transformation=trafo)
                    hqc = HamiltonianQC(parameters_qc)
                    Hmol=hqc.get_hamiltonian()
                    H=hqc()
                    if trafo=='JW':
                        self.assertTrue(parameters_qc.jordan_wigner())
                        self.assertEqual(H, openfermion.jordan_wigner(openfermion.get_fermion_operator(Hmol)))
                    else:
                        self.assertTrue(trafo=="BK")
                        self.assertTrue(parameters_qc.bravyi_kitaev())
                        self.assertEqual(H, openfermion.bravyi_kitaev(openfermion.get_fermion_operator(Hmol)))
