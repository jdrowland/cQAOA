import numpy as np
import cirq
import networkx as nx

def maxcut_hamiltonian(qubit_graph: nx.Graph) -> cirq.PauliSum:
    """Build a hamiltonian from the given qubit graph."""

    ham = cirq.PauliSum()
    for q1, q2 in qubit_graph.edges:
        ham += -1.0 * edge_operator(q1, q2)
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
