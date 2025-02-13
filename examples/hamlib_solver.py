from typing import Tuple
import argparse
import json
import numpy as np
import pandas as pd
import networkx as nx
import h5py
import cirq
import openfermion as of
from openfermion.transforms import qubit_operator_to_pauli_sum
from cqaoa.ansatz import CylicQAOAAnsatz
from cqaoa.hamlib_interface import print_hdf5_structure, read_graph_hdf5, read_openfermion_hdf5
from cqaoa.training import optimize_ansatz_random_start, cyclic_train
from cqaoa.maxcut import bitstrings_and_energies_from_df

def maxcut_hamiltonian_to_graph(hamiltonian: of.QubitOperator) -> nx.Graph:
    """Convert a given MaxCut Hamiltonian to a graph."""

    graph = nx.Graph()
    psum = qubit_operator_to_pauli_sum(hamiltonian)
    for pstring in psum:
        if len(pstring.qubits) == 2:
            graph.add_edge(pstring.qubits[0], pstring.qubits[1])
    return graph


def main() -> None:
    """Program entry point."""

    # Define CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON file for simulation input.")
    parser.add_argument("output_file", type=str, help="JSON file for simulation output.")
    args = parser.parse_args()

    # Parse input JSON.
    with open(args.input_file, "r", encoding="utf8") as f:
        input_dict = json.load(f)
    
    hamiltonian = read_openfermion_hdf5(input_dict["hdf_file"], input_dict["key"])
    hamiltonian_psum = qubit_operator_to_pauli_sum(hamiltonian)
    graph = maxcut_hamiltonian_to_graph(hamiltonian)

    # Solve with
    print("Solving with regular QAOA.")
    regular_ansatz = CylicQAOAAnsatz(graph, -1.0 * hamiltonian_psum)
    regular_result = optimize_ansatz_random_start(regular_ansatz, input_dict["p"], 10)
    regular_samples = regular_ansatz.sample_bitstrings(regular_result.gamma, regular_result.beta, input_dict["shots"])
    regular_bitstrings_energies = bitstrings_and_energies_from_df(regular_samples, -1.0 * hamiltonian_psum)
    regular_energies = [t[1] for t in regular_bitstrings_energies]
    regular_best_energy = min(regular_bitstrings_energies, key = lambda t: t[1])[1]
    print("Solving with cyclic QAOA.")
    p_cyclic = input_dict["p"] // input_dict["rounds"]
    cyclic_result = cyclic_train(graph, -1.0 * hamiltonian_psum, p_cyclic, input_dict["rounds"])

    # Serialize ouptut to JSON file.
    regular_dict = {
        "energy": regular_result.energy, 
        "sampled_energies": regular_energies,
        "best_energy": regular_best_energy,
        "gamma": list(regular_result.gamma), "beta": list(regular_result.beta)
    }
    cyclic_dict = {
        "energy_expectations": cyclic_result.energy_expectations, 
        "sampled_energies": cyclic_result.all_sampled_energies.tolist(),
        "lowest_sampled_energies": cyclic_result.lowest_sample_energy,
        "references": [list([bool(ri) for ri in r]) for r in cyclic_result.references],
        "gammas": [list(g) for g in cyclic_result.gammas], 
        "betas": [list(b) for b in cyclic_result.betas]
    }
    output_dict = {"input_filename": args.input_file, "input": input_dict, "regular_qaoa": regular_dict, "cyclic_qaoa": cyclic_dict}
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(output_dict, f)

if __name__ == "__main__":
    main()