import numpy as np

''' Program to write a matrix with the data of the new nuclei from the 
published paper, in matrix form.'''

Z = np.loadtxt('paper_data/Z.txt')
Z = Z[:, np.newaxis]
N = np.loadtxt('paper_data/N.txt')
N = N[:, np.newaxis]
DZBNN = np.loadtxt('paper_data/DZ-BNN.txt',delimiter='(')

paper_data = np.concatenate((N,Z,DZBNN), axis = 1)

np.savetxt('paper_data/published_data.txt', paper_data)