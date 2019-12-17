import numpy as np

Z = np.loadtxt('paper_data/Z.txt')
Z = Z[:, np.newaxis]
N = np.loadtxt('paper_data/N.txt')
N = N[:, np.newaxis]
DZBNN = np.loadtxt('paper_data/DZ-BNN.txt',delimiter='(')

paper_data = np.concatenate((Z,N,DZBNN), axis = 1)