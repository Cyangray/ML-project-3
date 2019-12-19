import numpy as np
import pickle

#import models
paper_data = np.loadtxt('paper_data/published_data.txt')
XGBoost_data = np.loadtxt('XGBoost_data.txt')
DecTrees_data = np.loadtxt('DecisionTree_data.txt')
FFNN_sigmoid_lin_data = np.loadtxt('FFNN_sigmoid_lin_data.txt')
FFNN_tanh_lin_data = np.loadtxt('FFNN_tanh_lin_data.txt')
FFNN_sigmoid_tanh_data = np.loadtxt('FFNN_sigmoid_tanh_data.txt')
FFNN_tanh_tanh_data = np.loadtxt('FFNN_tanh_tanh_data.txt')
Regression_OLS_data = np.loadtxt('Regression_OLS_data.txt')
Regression_ridge_data = np.loadtxt('Regression_ridge_data.txt')
Regression_LASSO_data = np.loadtxt('Regression_LASSO_data.txt')

# Load datasets
with open('datasets.pkl', 'rb') as input:
    Datasets = pickle.load(input)
AME12 = Datasets[0]
AME16 = Datasets[1]
testset = Datasets[2]

#Import experimental values (AME16)
neut = testset.df['N'].to_numpy()
prot = testset.df['Z'].to_numpy()
BAtest = testset.df['B'].to_numpy()
experimental_values = np.concatenate((neut[:,np.newaxis], prot[:,np.newaxis], BAtest[:,np.newaxis]), axis = 1)

#Keep only predicted elements also present in the paper
def keep_only_paper_elements(el_matrix):
    new_matrix = []
    for nucleus1 in paper_data:
        for nucleus2 in el_matrix:
            if (int(nucleus1[0]) == int(nucleus2[0])) and (int(nucleus1[1]) == int(nucleus2[1])):
                new_matrix.append(nucleus2)
                break
    return np.array(new_matrix)

reduced_XGB = keep_only_paper_elements(XGBoost_data)
reduced_DecTrees = keep_only_paper_elements(DecTrees_data)
reduced_FFNN_sigmoid_lin = keep_only_paper_elements(FFNN_sigmoid_lin_data)
reduced_FFNN_tanh_lin = keep_only_paper_elements(FFNN_tanh_lin_data)
reduced_FFNN_sigmoid_tanh = keep_only_paper_elements(FFNN_sigmoid_tanh_data)
reduced_FFNN_tanh_tanh = keep_only_paper_elements(FFNN_tanh_tanh_data)
reduced_regr_OLS = keep_only_paper_elements(Regression_OLS_data)
reduced_regr_ridge = keep_only_paper_elements(Regression_ridge_data)
reduced_regr_LASSO = keep_only_paper_elements(Regression_LASSO_data)

models = [paper_data, 
          reduced_XGB, 
          reduced_DecTrees, 
          reduced_FFNN_sigmoid_lin, 
          reduced_FFNN_tanh_lin,
          reduced_FFNN_sigmoid_tanh, 
          reduced_FFNN_tanh_tanh,
          reduced_regr_OLS,
          reduced_regr_ridge,
          reduced_regr_LASSO]

model_names = ['paper_data', 
              'reduced_XGB', 
               'reduced_DecTrees', 
               'reduced_FFNN_sigmoid_lin', 
               'reduced_FFNN_tanh_lin',
               'reduced_FFNN_sigmoid_tanh', 
               'reduced_FFNN_tanh_tanh',
               'reduced_regr_OLS',
               'reduced_regr_ridge',
               'reduced_regr_LASSO']

#compare model to experimental data
def evaluate_model(model):
    if np.array_equal(model, paper_data):
        '''Paper_data contains the difference between the DZ-BNN model 
        and the experimental value. We change the model to give the
        same quantity.'''
        diff = paper_data[:,2]
        comparison_list = 0
    else:
        comparison_list = []
        for model_nucleus in model:
            for exp_nucleus in experimental_values:
                if (int(model_nucleus[0]) == int(exp_nucleus[0])) and (int(model_nucleus[1]) == int(exp_nucleus[1])):
                    delta_B = model_nucleus[2] - exp_nucleus[2]
                    comparison_list.append([model_nucleus[0], model_nucleus[1], delta_B])
        comparison_list = np.array(comparison_list)
        diff = comparison_list[:,2]
    return comparison_list, np.std(diff)

comp_lists = []
for i, model in enumerate(models):
    complist, std = evaluate_model(model)
    comp_lists.append(complist)
    print(std, model_names[i])

        