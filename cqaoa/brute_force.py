import numpy as np
import math
from itertools import product
import networkx as nx
import cirq
from ising import *

#Classical Brute Force:
def generate_binary_strings(n):
    return [np.array(bits) for bits in product([0, 1], repeat=n)]
def brute_force(qubit_graph: nx.Graph, hamiltonian: cirq.PauliSum, verbose=False) -> (float, np.ndarray):
    """Brute force solve for the lowest energy bitstring"""
    minE = math.inf
    for string in generate_binary_strings(len(qubit_graph)):
        stringE = bitstring_energy(np.array(string), hamiltonian)
        if stringE  < minE:
            if verbose:
                print('New min E found! - ', stringE, ' with string: ', string)
            minE = stringE
            minString = string
    return minE, minString