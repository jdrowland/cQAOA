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

OptimizeResult = namedtuple("OptimizeResult", ["energy", "gamma", "beta"])

def optimize_ansatz(ansatz: CylicQAOAAnsatz, gamma: ndarray, beta) -> OptimizeResult:
    """Optimize the Ansatz for given starting values of gamma and beta."""

    def objective_callback(vars: np.ndarray):
        assert vars.size % 2 == 0
        
        gammas = vars[:(vars.size // 2)]
        betas = vars[(vars.size // 2):]
        return ansatz.energy(gammas, betas)

    vars0 = np.concatenate((gamma, beta))
    opt_result = minimize(objective_callback, vars0, method="Powell", options={"maxiter": 1_000_000})
    assert opt_result.success, f"Optimizer failed: {opt_result.message}"
    optimized_energy =  objective_callback(opt_result.x)
    gamma_opt = opt_result.x[:gamma.size]
    beta_opt = opt_result.x[gamma.size:]
    result = OptimizeResult(optimized_energy, gamma_opt, beta_opt)
    return result


def optimize_ansatz_random_start(ansatz: CylicQAOAAnsatz, layers: int, repetitions: int) -> OptimizeResult:
    """Optimize the Ansatz starting from an ensemble of starting values."""

    gammas = np.random.rand(repetitions, layers)
    betas = np.random.rand(repetitions, layers)
    all_outputs = []
    for i in range(repetitions):
        all_outputs.append(optimize_ansatz(ansatz, gammas[i, :], betas[i, :]))
    energies = [out[0] for out in all_outputs]
    i_opt = np.argmin(energies)
    return all_outputs[i_opt]


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
        energy_expectation, gamma, beta = optimize_ansatz_random_start(ansatz, p, random_starts)
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
