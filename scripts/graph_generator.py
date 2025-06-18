from typing import List
import argparse
import pickle
import numpy as np
import networkx as nx
import cirq

def random_lattice_qubit_graph(
    nrows: int, ncols: int, bz: float, gen: np.random.Generator,
    a: float = 1.0, b: float = 0.0
) -> nx.Graph:
    """Ising model with uniform longitudinal field and random interactions
    between neighboring spins on a 2D lattice. The bonds are random variables
    of the form j = ax + b, where x is in the uniform distribution over [0, 1]."""

    qs = cirq.GridQubit.rect(nrows, ncols)
    graph = nx.Graph()
    for i in range(nrows):
        for j in range(ncols):
            # Add bonds with neighbors.
            if i < j:
                if i != 0:
                    graph.add_edge(qs[i * ncols + j], qs[(i-1) * ncols + j], weight=a * gen.random() + b)
                if i != nrows - 1:
                    graph.add_edge(qs[i * ncols + j], qs[(i+1) * ncols + j], weight=a * gen.random() + b)
                if j != 0:
                    graph.add_edge(qs[i * ncols + j], qs[i * ncols + j-1], weight=a * gen.random() + b)
                if j != ncols - 1:
                    graph.add_edge(qs[i * ncols + j], qs[i * ncols + j+1], weight=a * gen.random() + b)
            # Add self-loop for the bz term.
            graph.add_edge(qs[i * ncols + j], qs[i * ncols + j], weight=bz)
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", type=str, help="pickle file for graph output.")
    parser.add_argument("rows", type=int, help="Number of rows.")
    parser.add_argument("cols", type=int, help="Number of cols.")
    parser.add_argument("--a", type=float, help="Multiplier for random bonds.", default=1.0)
    parser.add_argument("--b", type=float, help="Offset for random bonds.", default=0.0)
    parser.add_argument("--bz", type=float, help="Weight of self-loops in the graph.", default=1.0)
    args = parser.parse_args()

    gen = np.random.default_rng()
    graph = random_lattice_qubit_graph(args.rows, args.cols, args.bz, gen, a=args.a, b=args.b)
    with open(args.output_file, "wb") as f:
        pickle.dump(graph, f)

if __name__ == "__main__":
    main()
