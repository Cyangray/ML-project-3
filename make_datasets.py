import numpy as np
from dataset_objects import AtomicMasses
import matplotlib.pyplot as plt
import pickle
import copy

filepaths12 = ['data/mass12.txt', 'data/rct1-12.txt', 'data/rct2-12.txt']
filepaths16 = ['data/mass16.txt', 'data/rct1-16.txt', 'data/rct2-16.txt']

AME12 = AtomicMasses(filepaths12)
AME16 = AtomicMasses(filepaths16)

AME12.AMPolishDivide()
AME16.AMPolishDivide()

print(AME16.values[9,2])
#Normalize dataset.
AME16.normalize_dataset()
print(AME16.values[9,2])
#Normalize manually the other dataset so that they all have the same normalization
AME12.normalized = True
AME12.x_1d_unscaled = AME12.x_1d.copy()
AME12.y_1d_unscaled = AME12.y_1d.copy()
AME12.scaler = copy.deepcopy(AME16.scaler)
transformed_matrix = AME12.scaler.transform(AME12.values)
AME12.x_1d = transformed_matrix[:,:-1]
AME12.y_1d = transformed_matrix[:,-1]


testset = copy.deepcopy(AME16)

for idx, row in AME12.df.iterrows():
    testset.df.drop(idx, inplace = True)

Datasets = [AME12, AME16, testset]

#Save datasets
with open('datasets.pkl', 'wb') as output:
    pickle.dump(Datasets, output, pickle.HIGHEST_PROTOCOL)

