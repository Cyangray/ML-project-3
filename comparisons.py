import numpy as np
import pickle
import pandas as pd

#import models
paper_data = np.loadtxt('paper_data/published_data.txt')
XGBoost_data = np.loadtxt('XGBoost_data.txt')
DecTrees_data = np.loadtxt('DecisionTree_data.txt')

# Load datasets
with open('datasets.pkl', 'rb') as input:
    Datasets = pickle.load(input)

AME12 = Datasets[0]
AME16 = Datasets[1]
testset = Datasets[2]

#Import experimental values (AME16)
neut = testset.df['N'].to_numpy()
prot = testset.df['Z'].to_numpy()
BAtest = testset.df['B/A'].to_numpy()
experimental_values = np.concatenate((neut[:,np.newaxis], prot[:,np.newaxis], BAtest[:,np.newaxis]), axis = 1)

#Keep only predicted elements also present in the paper
def keep_only_paper_elements(el_matrix):
    new_matrix = []
    for nucleus1 in paper_data:
        for nucleus2 in el_matrix:
            if (int(nucleus1[0]) == int(nucleus2[0])) and (int(nucleus1[1]) == int(nucleus2[1])):
                new_matrix.append(nucleus2)
    return np.array(new_matrix)

reduced_XGB = keep_only_paper_elements(XGBoost_data)
reduced_DecTrees = keep_only_paper_elements(DecTrees_data)


models = [paper_data, reduced_XGB, reduced_DecTrees]

#compare model to experimental data
def evaluate_model(model):
    if np.array_equal(model, paper_data):
        diff = paper_data[:,2]
        comparison_list = 0
    else:
        comparison_list = []
        for nucleus1 in model:
            for nucleus2 in experimental_values:
                if (int(nucleus1[0]) == int(nucleus2[0])) and (int(nucleus1[1]) == int(nucleus2[1])):
                    '''Paper_data contains the difference between the DZ-BNN model 
                    and the experimental value. We change the model to give the
                    same quantity.'''
                    modeled_BA = nucleus1[2]*(nucleus1[0] + nucleus1[1]) - nucleus2[2]
                    comparison_list.append([nucleus1[0], nucleus1[1], modeled_BA, nucleus2[2]])
                        
        comparison_list = np.array(comparison_list)

        diff = comparison_list[:,2]# - comparison_list[:,3]
    return comparison_list, np.std(diff)

comp_lists = []
for model in models:
    complist, std = evaluate_model(model)
    comp_lists.append(complist)
    print(std)

        