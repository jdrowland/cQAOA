import unittest
import cirq
from cqaoa.maxcut import bitstring_energy

class TestEnergy(unittest.TestCase):

    def test_z_single_qubit(self):
        q = cirq.LineQubit(0)
        pstring = cirq.PauliString({q: cirq.Z})
        basis_state = [True]
        target_energy = -1.0
        real_energy = bitstring_energy(basis_state, pstring)
        self.assertAlmostEqual(real_energy, target_energy)

    def test_z_single_qubit_zero_state(self):
        q = cirq.LineQubit(0)
        pstring = cirq.PauliString({q: cirq.Z})
        basis_state = [False]
        target_energy = 1.0
        real_energy = bitstring_energy(basis_state, pstring)
        self.assertAlmostEqual(real_energy, target_energy)
    
    def test_single_qubit_x(self):
        q = cirq.LineQubit(0)
        pstring = cirq.PauliString({q: cirq.X})
        basis_state = [False]
        target_energy = 0.0
        real_energy = bitstring_energy(basis_state, pstring)
        self.assertAlmostEqual(real_energy, target_energy)
    
    def test_two_qubit_zz(self):
        qs = cirq.LineQubit.range(2)
        pstring = cirq.PauliString({qs[0]: cirq.Z, qs[1]: cirq.Z})
        basis_state = [False, True]
        target_energy = -1.0
        real_energy = bitstring_energy(basis_state, pstring)
        self.assertAlmostEqual(real_energy, target_energy)

    def test_two_qubit_iz(self):
        qs = cirq.LineQubit.range(2)
        pstring = cirq.PauliString({qs[1]: cirq.Z})
        basis_state = [True, False]
        target_energy = -1.0
        real_energy = bitstring_energy(basis_state, pstring)
        self.assertAlmostEqual(real_energy, target_energy)


if __name__ == "__main__":
    unittest.main()