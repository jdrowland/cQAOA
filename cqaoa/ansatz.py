from typing import Tuple, List
import numpy as np
import pandas as pd
import networkx as nx
import cirq
import qsimcirq

def gamma_layer(gamma: float, qubit_graph: nx.Graph, weighted: bool) -> cirq.Circuit:
    if weighted:
        gamma_ckt = cirq.Circuit()
        for q1, q2, w in qubit_graph.edges(data=True):
            if q1 == q2:
                gamma_ckt.append(cirq.Z(q1) ** (gamma*w['weight']))
            else:
                gamma_ckt.append(cirq.ZZ(q1, q2) ** (gamma*w['weight']))
        return gamma_ckt
    else:
        gamma_ckt = cirq.Circuit()
        for q1, q2 in qubit_graph.edges:
            #Enables Ising-like multigraphs:
            if q1 == q2:
                gamma_ckt.append(cirq.Z(q1) ** gamma)
            else:
                gamma_ckt.append(cirq.ZZ(q1, q2) ** gamma)
    return gamma_ckt


def beta_layer(beta: float, qubit_graph: nx.Graph) -> cirq.Circuit:
    beta_ckt = cirq.Circuit()
    for q in qubit_graph.nodes:
        beta_ckt.append(cirq.X(q) ** beta)
    return beta_ckt


def alpha_layer(alpha: float, qubit_graph: nx.Graph, reference: List[bool]) -> cirq.Circuit:
    alpha_ckt = cirq.Circuit()
    for i, q in enumerate(qubit_graph.nodes):
        if reference[i]:
            alpha_ckt.append(cirq.Z(q) ** alpha)
    return alpha_ckt


class CylicQAOAAnsatz:
    """Class to generate cyclic QAOA Ansatz circuits."""

    def __init__(self, qubit_graph, observable, weighted=False, **kwargs):
        """Initialize an instance of the CylicQAOA Ansatz.
        
        Arguments:
        qubit_graph: graph determining which qubits can interact.
        observable: observable for the problem (e.g. Hamiltonian for MaxCut)
        weighted: Enables the use of a weighted qubit graph
        
        Optional keyword arguments:
        reference: list of booleans, reference bitstring for the Ansatz.
        alpha: np.ndarray, coefficient for the refence term in this Hamiltonian.
        """

        self.qubit_graph = qubit_graph
        self.observable = observable
        self.weighted = weighted

        if not self.weighted and nx.is_weighted(qubit_graph):
            print("Weighted graph submitted without weighted=True flag while building the ansatz.  Did you mean to do this?")

        # Set up the cyclic part.
        if "reference" in kwargs.keys():
            self.use_reference = True
            self.reference = kwargs["reference"]
            if "alpha" in kwargs.keys():
                self.alpha = kwargs["alpha"]
            else:
                raise ValueError("alpha values must be provided with the reference.")
        else:
            self.use_reference = False
            if "alpha" in kwargs.keys():
                raise ValueError("Reference term must be provided alongside alpha.")
    
    @property
    def qubits(self):
        return self.qubit_graph.nodes
    
    def circuit(self, gammas: np.ndarray, betas: np.ndarray) -> cirq.Circuit:
        """Make a circuit with the given QAOA parameters. The graph determines
        the connectivity of the Ansatz."""

        assert gammas.size == betas.size

        qaoa_ckt = cirq.Circuit()
        for q in self.qubit_graph.nodes:
            qaoa_ckt.append(cirq.H(q))
        for i, (gamma, beta) in enumerate(zip(gammas, betas)):
            qaoa_ckt += gamma_layer(gamma, self.qubit_graph, self.weighted)
            qaoa_ckt += beta_layer(beta, self.qubit_graph)
            if self.use_reference:
                assert gammas.size == self.alpha.size, "Size of alpha and gamma must match."
                qaoa_ckt += alpha_layer(self.alpha[i], self.qubit_graph, self.reference)
        return qaoa_ckt
    
    def energy(self, gammas: np.ndarray, betas: np.ndarray) -> float:
        """Get the energy for this Ansatz with specific values."""

        qaoa_ckt = self.circuit(gammas, betas)
        sim = qsimcirq.QSimSimulator()
        try:
            energy = sim.simulate_expectation_values(qaoa_ckt, [self.observable])[0].real
            return energy
        except ValueError as exc:
            print(qaoa_ckt)
            qasm_str = cirq.qasm(qaoa_ckt)
            with open("circuit.qasm", "w", encoding="utf8") as f:
                f.write(qasm_str)
            raise exc
    
    def energy_sampled(self, gammas: np.ndarray, betas: np.ndarray, shots: int = 1000) -> float:
        """Get the energy via sampling for this Ansatz with specific values."""
        
        qaoa_ckt = self.circuit(gammas, betas)
        sim = cirq.Simulator()
        state = sim.simulate(qaoa_ckt).final_state_vector
        
        probs = np.abs(state)**2
        probs = probs / probs.sum()
        indices = np.random.choice(len(state), size=shots, p=probs)
        
        n = len(self.qubits)
        bitstrings = ((indices[:, None] >> np.arange(n)[::-1]) & 1).astype(np.int8)
        
        if not hasattr(self, '_energy_cache'):
            qs = list(self.qubit_graph.nodes())
            self._q_to_idx = {q: i for i, q in enumerate(qs)}
            self._edges = []
            self._local_fields = []
            for q1, q2, data in self.qubit_graph.edges(data=True):
                weight = data.get('weight', 1.0) if self.weighted else 1.0
                idx1 = self._q_to_idx[q1]
                idx2 = self._q_to_idx[q2]
                if q1 == q2:
                    self._local_fields.append((idx1, weight))
                else:
                    self._edges.append((idx1, idx2, weight))
            self._energy_cache = True
        
        z_values = 1 - 2 * bitstrings
        energies = np.zeros(shots)
        
        for (idx, weight) in self._local_fields:
            energies -= weight * z_values[:, idx]
        
        for (idx1, idx2, weight) in self._edges:
            energies -= weight * (-0.5 * z_values[:, idx1] * z_values[:, idx2] + 0.5)
        
        return np.mean(energies)

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
    
    def sample_bitstrings(self, gammas: np.ndarray, betas: np.ndarray, samples: int) -> pd.DataFrame:
        """Sample bitstrings from the final state generated by the Ansatz."""

        assert gammas.size == betas.size

        ckt = self.circuit(gammas, betas)
        # Append measurements to the end of the circuit.
        for q in ckt.all_qubits():
            ckt.append(cirq.M(q))
        sim = cirq.Simulator()
        # TODO Bug: the columns of the df are strings, so the qubits are not in the right order when we get a binary array.
        df = sim.sample(ckt, repetitions=samples)
        return df
