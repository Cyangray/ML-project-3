import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pickle

#Load the datasets
with open('datasets.pkl', 'rb') as input:
    Datasets = pickle.load(input)
AME12 = Datasets[0]
AME16 = Datasets[1]
testset = Datasets[2]

#Set the limits of the plot
maxN = int(AME16.df['N'].max()) + 1
maxZ = int(AME16.df['Z'].max()) + 1

def exctable(atomicdf, var = 'MassExcess', nans = True):
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
oldnuclei = exctable(AME12, var = 2, nans = False)
allnuclei = oldnuclei + newnuclei
allnuclei[allnuclei == 0] = np.nan

#Plot nuclei
plt.imshow(allnuclei.T, origin = 'lower', cmap = 'plasma')

#Draw magic numbers
Mnumbersx = np.array([2, 8, 20, 28, 50, 82, 126])
Mnumbersy = np.array([2, 8, 20, 28, 50, 82])
for x in Mnumbersx:
    for y in Mnumbersy:
        plt.axvline(x=x,color='black', alpha = 0.1)
        plt.axhline(y=y,color='black', alpha = 0.1)

plt.title('New nuclei in AME2016')
plt.xlabel('N')
plt.ylabel('Z')
plt.show()