from typing import Tuple
import numpy as np
import networkx as nx
import cirq

def gamma_layer(gamma: float, qubit_graph: nx.Graph) -> cirq.Circuit:
    gamma_ckt = cirq.Circuit()
    for q1, q2 in qubit_graph.edges:
        gamma_ckt.append(cirq.ZZ(q1, q2) ** gamma)
    return gamma_ckt


def beta_layer(beta: float, qubit_graph: nx.Graph) -> cirq.Circuit:
    beta_ckt = cirq.Circuit()
    for q in qubit_graph.nodes:
        beta_ckt.append(cirq.X(q) ** beta)
    return beta_ckt


def maxcut_hamiltonian(qubit_graph: nx.Graph) -> cirq.PauliSum:
    """Build a hamiltonian from the given qubit graph."""

    ham = cirq.PauliSum()
    for q1, q2 in qubit_graph.edges:
        ham += -1.0 * edge_operator(q1, q2)
    return ham


def edge_operator(q1: cirq.Qid, q2: cirq.Qid) -> cirq.PauliSum:
    """Operator from Eqn. 12 of QAOA paper. Used for the MaxCut Hamiltonian."""

    return -0.5 * cirq.PauliString({q1: cirq.Z, q2: cirq.Z}) + 0.5 * cirq.PauliString()


class CylicQAOAAnsatz:
    """Class to generate cyclic QAOA Ansatz circuits."""

    def __init__(self, qubit_graph, observable):
        self.qubit_graph = qubit_graph
        self.observable = observable
    
    def circuit(self, gammas: np.ndarray, betas: np.ndarray) -> cirq.Circuit:
        """Make a circuit with the given QAOA parameters. The graph determines
        the connectivity of the Ansatz."""

        assert gammas.size == betas.size

        qaoa_ckt = cirq.Circuit()
        for q in self.qubit_graph.nodes:
            qaoa_ckt.append(cirq.H(q))
        for gamma, beta in zip(gammas, betas):
            qaoa_ckt += gamma_layer(gamma, self.qubit_graph)
            qaoa_ckt += beta_layer(beta, self.qubit_graph)
        return qaoa_ckt
    
    def energy(self, gammas: np.ndarray, betas: np.ndarray) -> float:
        """Get the energy for this Ansatz with specific values."""

        qaoa_ckt = self.circuit(gammas, betas)
        sim = cirq.Simulator()
        return sim.simulate_expectation_values(qaoa_ckt, [self.observable])[0].real

    def energy_grad(self, gammas: np.ndarray, betas: np.ndarray, eps: float=1e-5) -> Tuple[np.ndarray, np.ndarray]:
        """Get the gradient for the given values of gamma and beta. This uses a finite difference."""

        gamma_grad = np.zeros(gammas.size)
        for i in range(gammas.size):
            gamma_plus = gammas.copy()
            gamma_plus[i] += eps
            gamma_minus = gammas.copy()
            gamma_minus[i] -= eps
            plus_energy = self.energy(gamma_plus, betas)
            minus_energy = self.energy(gamma_minus, betas)
            gamma_grad[i] = (plus_energy - minus_energy) / (2.0 * eps)
        beta_grad = np.zeros(betas.size)
        for i in range(betas.size):
            beta_plus = betas.copy()
            beta_plus[i] += eps
            beta_minus = betas.copy()
            beta_minus[i] -= eps
            plus_energy = self.energy(gammas, beta_plus)
            minus_energy = self.energy(gammas, beta_minus)
            beta_grad[i] = (plus_energy - minus_energy) / (2.0 * eps)
        return gamma_grad, beta_grad
