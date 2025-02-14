import numpy as np
import cirq
import networkx as nx

def ising_hamiltonian(qubit_graph: nx.Graph, weighted=False) -> cirq.PauliSum:
    """Build an ising hamiltonian from the given qubit graph."""
    
    if not weighted and nx.is_weighted(qubit_graph):
            print("Weighted graph submitted without weighted=True flag while building the Hamiltonian.  Did you mean to do this?")
    ham = cirq.PauliSum()
    if weighted:
        for q1, q2, w in qubit_graph.edges(data=True):
            if q1 == q2:
                ham -= w['weight']* cirq.PauliString({q1:cirq.Z})
            else:
                ham -= w['weight'] * edge_operator(q1, q2)
    else:
        for q1, q2 in qubit_graph.edges():
            if q1 == q2:
                ham -= cirq.PauliString({q1:cirq.Z})
            else:
                ham -= edge_operator(q1, q2)
    return ham


def edge_operator(q1: cirq.Qid, q2: cirq.Qid) -> cirq.PauliSum:
    """Operator from Eqn. 12 of QAOA paper. Used for the MaxCut Hamiltonian."""

    return -0.5 * cirq.PauliString({q1: cirq.Z, q2: cirq.Z}) + 0.5 * cirq.PauliString()


def bitstring_energy(bits: np.ndarray, hamiltonian: cirq.PauliSum) -> float:
    """Get the energy for a given bitstring."""

    # Build a circuit that prepares the given computational basis state.
    ckt = cirq.Circuit()
    qs = hamiltonian.qubits
    for q, bit in zip(qs, bits):
        if bit:
            ckt.append(cirq.X(q))
        else:
            ckt.append(cirq.I(q))
    # Compute the energy expectation value for this circuit (and thus the state).
    sim = cirq.Simulator()
    expectation_values = sim.simulate_expectation_values(ckt, [hamiltonian])
    return expectation_values[0].real
    
def circuit(self, gammas: np.ndarray, betas: np.ndarray) -> cirq.Circuit:
        """Make a circuit with the given QAOA parameters. The graph determines
        the connectivity of the Ansatz."""

        assert gammas.size == betas.size

        qaoa_ckt = cirq.Circuit()
        for q in self.qubit_graph.nodes:
            qaoa_ckt.append(cirq.H(q))
        for i, (gamma, beta) in enumerate(zip(gammas, betas)):
            qaoa_ckt += gamma_layerw(gamma, self.qubit_graph)
            qaoa_ckt += beta_layer(beta, self.qubit_graph)
            if self.use_reference:
                assert gammas.size == self.alpha.size, "Size of alpha and gamma must match."
                qaoa_ckt += alpha_layer(self.alpha[i], self.qubit_graph, self.reference)
        return qaoa_ckt