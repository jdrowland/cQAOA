from typing import Tuple, List
from collections import namedtuple
import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
import cirq
import networkx as nx
try:
    from .ansatz import CylicQAOAAnsatz
    from .maxcut import bitstring_energy, bitstrings_and_energies_from_df
except:
    from ansatz import CylicQAOAAnsatz
    from maxcut import bitstring_energy, bitstrings_and_energies_from_df

OptimizeResult = namedtuple("OptimizeResult", ["energy", "gamma", "beta", "best_sampled_energy", "best_sampled_string"])

def optimize_ansatz(ansatz: CylicQAOAAnsatz, gamma: ndarray, beta, sampled=False, shots=1000) -> OptimizeResult:
    """Optimize the Ansatz for given starting values of gamma and beta."""

    def objective_callback(vars: np.ndarray):
        assert vars.size % 2 == 0
        
        gammas = vars[:(vars.size // 2)]
        betas = vars[(vars.size // 2):]
        if sampled:
            return ansatz.energy_sampled(gammas, betas, shots)
        else:
            return ansatz.energy(gammas, betas)

    vars0 = np.concatenate((gamma, beta))
    opt_result = minimize(objective_callback, vars0, method="Powell", options={"maxiter": 1_000_000})
    assert opt_result.success, f"Optimizer failed: {opt_result.message}"
    optimized_energy =  objective_callback(opt_result.x)
    gamma_opt = opt_result.x[:gamma.size]
    beta_opt = opt_result.x[gamma.size:]
    result = OptimizeResult(optimized_energy, gamma_opt, beta_opt, None, None)
    return result


def optimize_ansatz_random_start(ansatz: CylicQAOAAnsatz, layers: int, repetitions: int, shots=1000, sampled=False):
    """Optimize the Ansatz starting from an ensemble of starting values."""

    gammas = np.random.rand(repetitions, layers)
    betas = np.random.rand(repetitions, layers)
    all_outputs = []
    for i in range(repetitions):
        all_outputs.append(optimize_ansatz(ansatz, gammas[i, :], betas[i, :], sampled, shots))
    energies = [out[0] for out in all_outputs]
    i_opt = np.argmin(energies)
    
    best_result = all_outputs[i_opt]
    
    # Sample once to find best energy and bitstring
    sim = cirq.Simulator()
    state = sim.simulate(ansatz.circuit(best_result.gamma, best_result.beta)).final_state_vector
    probs = np.abs(state)**2
    probs = probs / probs.sum()
    sampled_indices = np.random.choice(len(state), p=probs, size=shots)
    from .ising import bitstring_energy
    
    best_sampled_energy = float('inf')
    best_sampled_string = None
    for idx in sampled_indices:
        bitstring = np.array(list(map(int, f"{idx:0{len(ansatz.qubits)}b}")))
        energy = bitstring_energy(bitstring, ansatz.observable)
        if energy < best_sampled_energy:
            best_sampled_energy = energy
            best_sampled_string = bitstring
    
    # Return the full OptimizeResult with sampling info
    return OptimizeResult(best_result.energy, best_result.gamma, best_result.beta, best_sampled_energy, best_sampled_string)


CyclicResult = namedtuple("CyclicResult", ["ansatz", "energy_expectations", "all_sampled_energies", "lowest_sample_energy", "references", "gammas", "betas"])


def cyclic_train(
    qubit_graph: nx.Graph, hamiltonian: cirq.PauliSum, p: int, rounds: int,
    alpha0: float=2.0, shots: int=1000, random_starts: int=10, weighted: bool=False
) -> CyclicResult:
    """Train by repeatedly taking the lowest-energy string as the reference
    for the next round of training."""

    reference = [True] * len(qubit_graph.nodes)
    alpha = np.linspace(alpha0, 0.0, num=p)
    bitstrings = []
    energies = []
    lowest_sampled_energies = []
    all_sampled_energies = np.zeros((shots, rounds), dtype=float)
    gammas = []
    betas = []
    for i in range(rounds):
        print(f"Cyclic QAOA round {i+1} of {rounds}.")
        # Get the energy of the current reference.
        old_reference_energy = bitstring_energy(reference, hamiltonian)
        # Train the Ansatz to minimize expectation values.
        ansatz = CylicQAOAAnsatz(qubit_graph, hamiltonian ,weighted=weighted, reference=reference, alpha=alpha)
        result = optimize_ansatz_random_start(ansatz, p, random_starts)
        energy_expectation = result.energy
        gamma = result.gamma
        beta = result.beta
        # Sample bitstrings and energies to choose the lowest one.
        sample_df = ansatz.sample_bitstrings(gamma, beta, shots)
        sample_bitstrings_energies = bitstrings_and_energies_from_df(sample_df, hamiltonian)
        this_round_energies = [t[1] for t in sample_bitstrings_energies]
        sampled_bitstrings = [t[0] for t in sample_bitstrings_energies]
        i_best = np.argmin(this_round_energies)
        bitstrings.append(reference)
        energies.append(energy_expectation)
        lowest_sampled_energies.append(this_round_energies[i_best])
        all_sampled_energies[:, i] = this_round_energies
        gammas.append(gamma)
        betas.append(beta)
        # Only set the reference to the new best value is it is the lowest one seen.
        if this_round_energies[i_best] <= old_reference_energy:
            reference = sampled_bitstrings[i_best]
    result = CyclicResult(ansatz, energies, all_sampled_energies, lowest_sampled_energies, bitstrings, gammas, betas)
    return result


# New helper functions added at the end to minimize diffs
def compute_all_energies(hamiltonian: cirq.PauliSum) -> ndarray:
    """Compute energies for all possible bitstrings. Index i corresponds to bitstring bin(i)."""
    
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
    
    n_states = 2**n_qubits
    energies = np.zeros(n_states)
    
    for idx in range(n_states):
        z_values = np.zeros(n_qubits)
        for i in range(n_qubits):
            z_values[i] = 1 - 2 * ((idx >> i) & 1)
        
        energy = constant
        for i, coeff in z_terms:
            energy += coeff * z_values[i]
        for i, j, coeff in zz_terms:
            energy += coeff * z_values[i] * z_values[j]
        energies[idx] = energy
    
    return energies


def compute_ground_state_probability(ansatz: CylicQAOAAnsatz, gamma: ndarray, beta: ndarray) -> float:
    """Compute the probability of measuring the ground state for given gamma and beta parameters."""
    from .brute_force import brute_force2
    
    sim = cirq.Simulator()
    state = sim.simulate(ansatz.circuit(gamma, beta)).final_state_vector
    
    ground_energy, ground_string = brute_force2(ansatz.observable)
    ground_idx = int(''.join(str(b) for b in ground_string), 2)
    ground_prob = float(np.abs(state[ground_idx])**2)
    
    return ground_prob


def compute_improvement_probability(ansatz: CylicQAOAAnsatz, gamma: ndarray, beta: ndarray, reference_energy: float) -> float:
    """Compute the probability of measuring any state with energy lower than reference_energy."""
    
    sim = cirq.Simulator()
    state = sim.simulate(ansatz.circuit(gamma, beta)).final_state_vector
    probs = np.abs(state)**2
    
    energies = compute_all_energies(ansatz.observable)
    
    improvement_prob = np.sum(probs[energies < reference_energy])
    
    return improvement_prob


# Wrapper functions for backward compatibility
def optimize_ansatz_sampled(ansatz: CylicQAOAAnsatz, gamma: ndarray, beta: ndarray, shots: int = 1000) -> OptimizeResult:
    """Optimize the Ansatz using sampled energy estimates."""
    return optimize_ansatz(ansatz, gamma, beta, sampled=True, shots=shots)


def optimize_ansatz_random_start_sampled(ansatz: CylicQAOAAnsatz, layers: int, repetitions: int, shots: int = 1000):
    """Optimize the Ansatz starting from an ensemble of starting values using sampled energy."""
    
    result = optimize_ansatz_random_start(ansatz, layers, repetitions, sampled=True, shots=shots)
    
    ground_prob = compute_ground_state_probability(ansatz, result.gamma, result.beta)
    
    return result.energy, result.gamma, result.beta, result.best_sampled_energy, ground_prob, result.best_sampled_string