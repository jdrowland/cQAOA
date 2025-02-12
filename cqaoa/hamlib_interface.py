# nb code snippets from https://arxiv.org/abs/2306.13126

import networkx as nx
import mat2qubit as m2q
import openfermion as of
import h5py
import numpy as np

def parse_through_hdf5(func):
    """ 
    Decorator function that iterates through an HDF5 file and performs
    the action specified by ‘ func ‘ on the internal and leaf nodes in the HDF5 file. 
    """

    def wrapper (obj, path = '/', key = None) :
        if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
            for ky in obj.keys() :
                func(obj, path, key=ky, leaf = False)
                wrapper(obj = obj[ky], path = path + ky + ',', key = ky)
        elif type (obj) == h5py._hl.dataset.Dataset:
            func(obj, path, key = None, leaf = True)
    return wrapper


def print_hdf5_structure ( fname_hdf5 : str ) :
    """ 
    Print the path structure of the HDF5 file.
    Args
    ----
    fname_hdf5 ( str ) : full path where HDF5 file is stored
    """

    @parse_through_hdf5
    def action(obj, path = '/', key = None, leaf = False):
        if key is not None :
            print((path.count('/') -1) * '\t' , '-', key, ':', path + key + '/')
        if leaf:
            print((path.count('/') -1) * '\t' , '[^^ DATASET ^^]')
    
    with h5py.File(fname_hdf5 , 'r') as f:
        action(f['/'])


def get_hdf5_keys ( fname_hdf5 : str ) :
    """ Get a list of keys to all datasets stored in the HDF5 file .
    Args
    ----
    fname_hdf5 ( str ) : full path where HDF5 file is stored
    """

    all_keys = []
    @parse_through_hdf5
    def action(obj, path = '/', key = None, leaf = False):
        if leaf is True :
            all_keys.append(path)
    
    with h5py.File(fname_hdf5, 'r') as f:
        action(f['/'])
    return all_keys


def read_graph_hdf5(fname_hdf5: str, key: str):
    """ 
    Read networkx graphs from HDF5 file at specified key . Returns a single networkx
    graph.
    """

    with h5py.File(fname_hdf5, 'r') as f:
        G = nx.Graph(list(np.array(f[key])))
    return G


def read_gridpositions_hdf5(fname_hdf5: str , key: str ):
    """ 
    Read grid positions, stored as attribute of each networkx graph from HDF5 file
    at specified key. Returns grid positions of nodes associated with a single graph.
    """

    with h5py.File(fname_hdf5 , 'r') as f:
        dataset = f[key]
        gridpositions_dict = dict(dataset.attrs.items())
    return gridpositions_dict


def read_openfermion_hdf5(fname_hdf5: str, key: str, optype=of.QubitOperator):
    """ 
    Read any openfermion operator object from HDF5 file at specified key.
    'optype' is the op class, can be of.QubitOperator or of.FermionOperator.
    """

    with h5py.File(fname_hdf5, 'r', libver='latest') as f:
        op = optype(f[key][()].decode("utf-8"))
    return op


def read_mat2qubit_hdf5(fname_hdf5: str , key: str):
    """ 
    Returns mat2qubit ’s qSymbOp operator from HDF5 file at specified key . 
    """

    with h5py.File(fname_hdf5 , 'r') as f:
        op = m2q.qSymbOp(f[key][()].decode("utf-8"))
    return op


def read_clause_list_hdf5(fname_hdf5: str, key: str ):
    """ 
    Read clause list from HDF5 file at specified key. Returns clause list in DIMACS
    format. 
    """

    clause_list = []
    with h5py.File(fname_hdf5, 'r') as f:
        for clause in list(np.array(f[key])):
            clause_list . append ([ v for v in clause ])
    return clause_list