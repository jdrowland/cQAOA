from typing import Tuple, List
import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
import cirq
import networkx as nx
from ansatz import CylicQAOAAnsatz
from maxcut import bitstring_energy

def optimize_ansatz(ansatz: CylicQAOAAnsatz, gamma: ndarray, beta) -> Tuple[float, ndarray, ndarray]:
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
    return optimized_energy, gamma_opt, beta_opt


def optimize_ansatz_random_start(ansatz: CylicQAOAAnsatz, layers: int, repetitions: int) -> Tuple[float, ndarray, ndarray]:
    """Optimize the Ansatz starting from an ensemble of starting values."""

    gammas = np.random.rand(repetitions, layers)
    betas = np.random.rand(repetitions, layers)
    all_outputs = []
    for i in range(repetitions):
        all_outputs.append(optimize_ansatz(ansatz, gammas[i, :], betas[i, :]))
    energies = [out[0] for out in all_outputs]
    i_opt = np.argmin(energies)
    return all_outputs[i_opt]


def cyclic_train(qubit_graph: nx.Graph, hamiltonian: cirq.PauliSum, p: int, rounds: int) -> Tuple[List[float], List[ndarray], List[ndarray], List[ndarray]]:
    """Train by repeatedly taking the lowest-energy string as the reference
    for the next round of training."""

    reference = [True] * len(qubit_graph.nodes)
    alpha = np.linspace(2.0, 0.0, num=p)
    bitstrings = []
    energies = []
    gammas = []
    betas = []
    for i in range(rounds):
        ansatz = CylicQAOAAnsatz(qubit_graph, hamiltonian, reference=reference, alpha=alpha)
        _, gamma, beta = optimize_ansatz_random_start(ansatz, p, 10)
        sampled_bitstrings = ansatz.sample_bitstrings(gamma, beta, 1000)
        this_round_energies = [bitstring_energy(sampled_bitstrings[i, :], hamiltonian) for i in range(sampled_bitstrings.shape[0])]
        i_best = np.argmin(this_round_energies)
        bitstrings.append(reference)
        energies.append(this_round_energies[i_best])
        gammas.append(gamma)
        betas.append(beta)
        # Only set the reference to the new best value is it is the lowest one seen.
        if this_round_energies[i_best] or i == 0:
            reference = sampled_bitstrings[i_best, :]
    return energies, bitstrings, gammas, betas