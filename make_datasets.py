import numpy as np
from dataset_objects import AtomicMasses
import matplotlib.pyplot as plt
import pickle

filepaths12 = ['data/mass12.txt', 'data/rct1-12.txt', 'data/rct2-12.txt']
filepaths16 = ['data/mass16.txt', 'data/rct1-16.txt', 'data/rct2-16.txt']

Excess12 = AtomicMasses(filepaths12)
Excess16 = AtomicMasses(filepaths16)
testset = AtomicMasses(filepaths16)

Excess12.AMPolishDivide()
Excess16.AMPolishDivide()
testset.AMPolishDivide()

for idx, row in Excess12.df.iterrows():
    testset.df.drop(idx, inplace = True)

Datasets = [Excess12, Excess16, testset]

#Save datasets
with open('datasets.pkl', 'wb') as output:
    pickle.dump(Datasets, output, pickle.HIGHEST_PROTOCOL)

