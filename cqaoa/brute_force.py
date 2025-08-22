import numpy as np
import math
from itertools import product
import networkx as nx
import cirq
try:
    from .ising import *
except ImportError:
    from ising import *

def generate_binary_strings(n):
    for bits in product([0, 1], repeat=n):
        yield np.array(bits, dtype=np.int8)

def brute_force(hamiltonian: cirq.PauliSum, verbose=False) -> (float, np.ndarray):
    """Brute force solve for the lowest energy bitstring"""
    # Get number of qubits from the hamiltonian
    qubits = hamiltonian.qubits
    n_qubits = len(qubits)
    
    minE = math.inf
    minString = None
    
    for string in generate_binary_strings(n_qubits):
        stringE = bitstring_energy(string, hamiltonian)
        if stringE < minE:
            if verbose:
                print('New min E found! - ', stringE, ' with string: ', string)
            minE = stringE
            minString = string.copy()
    
    return minE, minString

def brute_force2(hamiltonian: cirq.PauliSum, verbose=False) -> (float, np.ndarray):
    """Optimized brute force using pre-processed Hamiltonian terms"""
    qubits = sorted(hamiltonian.qubits)
    n_qubits = len(qubits)
    qubit_to_idx = {q: i for i, q in enumerate(qubits)}
    
    z_terms = []
    zz_terms = []
    constant = 0.0
    
    for term in hamiltonian:
        pauli_dict = term._qubit_pauli_map
        coefficient = term.coefficient
        
        z_qubits = [q for q, op in pauli_dict.items() if op == cirq.Z]
        
        if len(z_qubits) == 0:
            constant += coefficient.real
        elif len(z_qubits) == 1:
            idx = qubit_to_idx[z_qubits[0]]
            z_terms.append((idx, coefficient.real))
        elif len(z_qubits) == 2:
            idx1 = qubit_to_idx[z_qubits[0]]
            idx2 = qubit_to_idx[z_qubits[1]]
            zz_terms.append((idx1, idx2, coefficient.real))
    
    minE = math.inf
    minString = None
    
    for string in generate_binary_strings(n_qubits):
        z_values = 1 - 2 * string
        
        energy = constant
        for idx, coeff in z_terms:
            energy += coeff * z_values[idx]
        for idx1, idx2, coeff in zz_terms:
            energy += coeff * z_values[idx1] * z_values[idx2]
        
        if energy < minE:
            if verbose:
                print('New min E found! - ', energy, ' with string: ', string)
            minE = energy
            minString = string.copy()
    
    return minE, minString