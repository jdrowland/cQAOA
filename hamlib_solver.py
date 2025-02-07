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
from ansatz import CylicQAOAAnsatz
from hamlib_interface import print_hdf5_structure, read_graph_hdf5, read_openfermion_hdf5
from training import optimize_ansatz_random_start, cyclic_train
from maxcut import bitstring_energy

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
    regular_ansatz = CylicQAOAAnsatz(graph, -1.0 * hamiltonian_psum)
    regular_result = optimize_ansatz_random_start(regular_ansatz, input_dict["p"], 10)
    p_cyclic = input_dict["p"] // input_dict["rounds"]
    cyclic_result = cyclic_train(graph, -1.0 * hamiltonian_psum, p_cyclic, input_dict["rounds"])

    # Serialize ouptut to JSON file.
    regular_dict = {
        "energy": regular_result.energy, 
        "gamma": list(regular_result.gamma), "beta": list(regular_result.beta)
    }
    # TODO save alpha as well
    cyclic_dict = {
        "energies": cyclic_result.energies, 
        "references": [list([bool(ri) for ri in r]) for r in cyclic_result.references],
        "gammas": [list(g) for g in cyclic_result.gammas], 
        "betas": [list(b) for b in cyclic_result.betas]
    }
    output_dict = {"input": input_dict, "regular_qaoa": regular_dict, "cyclic_qaoa": cyclic_dict}
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(output_dict, f)

if __name__ == "__main__":
    main()