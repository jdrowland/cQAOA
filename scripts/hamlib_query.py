import sys
import argparse
sys.path.append("..")
import hamlib_interface as hi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type="str", help="HDF5 file to query.")
    args = parser.parse_args()

    keys = hi.get_hdf5_keys(args.filename)
    for key in keys:
        print(key)

if __name__ == "__main__":
    main()