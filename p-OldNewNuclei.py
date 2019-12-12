import numpy as np
from dataset_objects import AtomicMasses
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pickle

#Load the datasets
with open('datasets.pkl', 'rb') as input:
    Datasets = pickle.load(input)

Excess12 = Datasets[0]
Excess16 = Datasets[1]
testset = Datasets[2]

def exctable(atomicdf, var = 'MassExcess', nans = True):
    maxN = 156
    maxZ = 103
    
    table = np.zeros((maxN,maxZ,))
    if nans:
        table[:] = np.nan
    
    if isinstance(var, str):
        for idx, row in atomicdf.df.iterrows():
            table[int(row['N']), int(row['Z'])] = row[var]
    else:    
        for idx, row in atomicdf.df.iterrows():
            table[int(row['N']), int(row['Z'])] = var
    return table

newnuclei = exctable(testset, var = 1, nans = False)
oldnuclei = exctable(Excess12, var = 2, nans = False)
allnuclei = oldnuclei + newnuclei
allnuclei[allnuclei == 0] = np.nan

plt.imshow(allnuclei.T, origin = 'lower', cmap = 'plasma')
plt.colorbar()
plt.show()