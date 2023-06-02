import sys 
import pickle
import openfermion as of 
import time 

start = time.time() 
mol = sys.argv[1]
fname = mol + '_fer.bin'

with open(fname, 'rb') as f:
    Hf = pickle.load(f)

gs_e, gs = of.get_ground_state(of.get_sparse_operator(Hf))
print("Mol: {}".format(mol.upper()))
print("GS energy: {}".format(gs_e))

time_sec = time.time() - start
print("Time elapsed: {} min".format(round(time_sec/60, 1)))
