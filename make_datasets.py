from dataset_objects import AtomicMasses
from visualization import draw_dataset
import pickle
import copy

''' Program that makes the datasets of the Atomic Masses Estimates to be used
in the project '''

#Import datasets from file and add to objects
filepaths12 = ['data/mass12.txt', 'data/rct1-12.txt', 'data/rct2-12.txt']
filepaths16 = ['data/mass16.txt', 'data/rct1-16.txt', 'data/rct2-16.txt']
AME12 = AtomicMasses(filepaths12, usealldata = False)
AME16 = AtomicMasses(filepaths16, usealldata = False)

#Polish the dataset and divide into features and target
AME12.AMPolishDivide(total_binding_energy = True)
AME16.AMPolishDivide(total_binding_energy = True)

draw_dataset(AME16, 'B')

#Normalize dataset.
AME16.normalize_dataset()

#Normalize manually the other dataset so that they all have the same normalization
AME12.normalized = True
AME12.x_1d_unscaled = AME12.x_1d.copy()
AME12.y_1d_unscaled = AME12.y_1d.copy()
AME12.scaler = copy.deepcopy(AME16.scaler)
transformed_matrix = AME12.scaler.transform(AME12.values)
AME12.x_1d = transformed_matrix[:,:-1]
AME12.y_1d = transformed_matrix[:,-1]

#Make dataset for only the new nuclei found in AME16 and not in AME12
testset = copy.deepcopy(AME16)
for idx, row in AME12.df.iterrows():
    testset.df.drop(idx, inplace = True)

#Save datasets to file
Datasets = [AME12, AME16, testset]
with open('datasets.pkl', 'wb') as output:
    pickle.dump(Datasets, output, pickle.HIGHEST_PROTOCOL)

