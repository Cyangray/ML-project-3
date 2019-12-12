import numpy as np
from dataset_objects import AtomicMasses
import matplotlib.pyplot as plt

filepaths12 = ['data/mass12.txt', 'data/rct1-12.txt', 'data/rct2-12.txt']
filepaths16 = ['data/mass16.txt', 'data/rct1-16.txt', 'data/rct2-16.txt']

Excess12 = AtomicMasses(filepaths12)
Excess16 = AtomicMasses(filepaths16)

Excess12.AMPolishDivide()
Excess16.AMPolishDivide()

def exctable(atomicdf):
    maxN = atomicdf.df['N'].max() + 1
    maxZ = atomicdf.df['Z'].max() + 1
    N = np.array(range(maxN))
    Z = np.array(range(maxZ))
    
    table = np.empty((maxN,maxZ,))
    table[:] = np.nan
    
    for idx, row in atomicdf.df.iterrows():
        table[int(row['N']), int(row['Z'])] = row['MassExcess']
    return table

table1 = exctable(Excess12)
table2 = exctable(Excess16)

diff = table2 - table1

plt.imshow(diff.T, origin = 'lower', cmap = 'plasma')
plt.colorbar()
plt.show()



