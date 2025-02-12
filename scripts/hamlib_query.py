import sys
import argparse
import cqaoa.hamlib_interface as hi
from openfermion.utils import count_qubits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="HDF5 file to query.")
    args = parser.parse_args()

    keys = hi.get_hdf5_keys(args.filename)
    print("key,terms,qubits")
    for key in keys:
        trimmed_key = key[1:-1]
        ham = hi.read_openfermion_hdf5(args.filename, trimmed_key)
        terms = len(ham.terms)
        qubits = count_qubits(ham)
        print(f"{trimmed_key},{terms},{qubits}")

if __name__ == "__main__":
    main()