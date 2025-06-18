from typing import Tuple
import os
import argparse
import pickle
import json
import numpy as np
import networkx as nx
import cirq
import openfermion as of
from openfermion.transforms import qubit_operator_to_pauli_sum
from cqaoa.ansatz import CylicQAOAAnsatz
from cqaoa.hamlib_interface import print_hdf5_structure, read_graph_hdf5, read_openfermion_hdf5
from cqaoa.training import optimize_ansatz_random_start, cyclic_train
from cqaoa.maxcut import bitstrings_and_energies_from_df
from cqaoa.ising import ising_hamiltonian


def solve_regular(ansatz: CylicQAOAAnsatz, p: int, shots: int, reps: int=10):
    """Solve problem with regular QAOA."""

    energy_expectation, gamma, beta = optimize_ansatz_random_start(ansatz, p)
    samples_df = ansatz.sample_bitstrings(gamma, beta, shots)
    bitstrings_and_energies = bitstrings_and_energies_from_df(samples_df)
    return energy_expectation, gamma, beta, bitstrings_and_energies


def main():
    parser = argparse.ArgumentParser(
        prog="ising_solver",
        description="Find ground state of random bond ising model by both cyclic and regular QAOA."
    )
    parser.add_argument("input_file", type=str, help="JSON input file in data_dir.")
    parser.add_argument("output_file", type=str, help="JSON output file in data_dir.")
    parser.add_argument("--data_dir", type=str,
                        help="Directory to look for graph files and io files.",
                        default=".")
    args = parser.parse_args()

    # Read input file and get problem graph.
    print("Solving with regular QAOA.")
    with open(os.path.join(args.data_dir, args.input_file), "r", encoding="utf8") as f:
        input_dict = json.load(f)
    with open(os.path.join(args.data_dir, input_dict["graph_file"]), "rb") as f:
        graph = pickle.load(f)
    
    assert input_dict["p"] % input_dict["rounds"] == 0, "p must be divisible by number of rounds."
    
    hamiltonian = ising_hamiltonian(graph, weighted=True)
    ansatz = CylicQAOAAnsatz(graph, hamiltonian, weighted=True)

    # Solve by regular QAOA.
    regular_result = optimize_ansatz_random_start(ansatz, input_dict["p"], repetitions=10)
    samples_df = ansatz.sample_bitstrings(regular_result.gamma, regular_result.beta, input_dict["shots"])
    bitstrings_and_energies = bitstrings_and_energies_from_df(samples_df)
    regular_energies = [t[1] for t in bitstrings_and_energies]
    regular_best_index = np.argmin(regular_energies)
    regular_best_energy = bitstrings_and_energies[regular_best_index][1]
    regular_bitstring = bitstrings_and_energies[regular_best_index][0]
    regular_result_dict = {
        "energy_expectation": regular_result.energy,
        "gamma": regular_result.gamma,
        "beta": regular_result.beta,
        "energies": regular_energies,
        "lowest_energy": regular_best_energy,
        "best_bitstring": regular_bitstring
    }

    # Solve by cyclic QAOA.
    print("Solving with cyclic QAOA.")
    p_cyclic = input_dict["p"] // input_dict["rounds"]
    cyclic_result = cyclic_train(
        graph, hamiltonian, p_cyclic, input_dict["rounds"],
        alpha0=input_dict["alpha0"], shots=input_dict["shots"]
    )

    results_dict = {
        "input_file": args.input_file,
        "input": input_dict,
        "regular_qaoa": regular_result_dict,
        "cyclic_qaoa": cyclic_result
    }
    with open(os.path.join(args.data_dir, args.output_file), "w") as f:
        json.dump(results_dict, f)

if __name__ == "__main__":
    main()
