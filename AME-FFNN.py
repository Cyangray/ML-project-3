import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from visualization import plot_features, plot_correlation_matrix, show_heatmap_mse_R2, plot_3d_terrain
from dataset_objects import dataset, credit_card_dataset
from fit_matrix import fit
import statistical_functions as statistics
from sampling_methods import sampling
from sklearn.metrics import roc_auc_score
from sklearn import datasets
from functions import make_onehot, inverse_onehot
from neural_network import NeuralNetwork, layer
import seaborn as sns
import pickle

#k-fold cross validation parameters
CV = False
k = 5
np.random.seed(1234)

#Stochastic gradient descent parameters
m = 20           #Number of minibatches
Niterations = 5000
deg = 0


# Load dataset
with open('datasets.pkl', 'rb') as input:
    Datasets = pickle.load(input)

AME12 = Datasets[0]
AME16 = Datasets[1]
testset = Datasets[2]
    
#Divide in train and test
AME16.sort_train_test(AME12, useAME12 = False)

#Make model
AME16Fit = fit(AME16)

#Create polynomial design matrix for train and test sets
X_train = AME16Fit.create_design_matrix(deg = deg)
X_test = AME16Fit.create_design_matrix(x = AME16.test_x_1d, deg = deg)

#Initialize inputs for Neural Network
y_train = AME16.y_1d[:,np.newaxis]
y_test = AME16.test_y_1d[:,np.newaxis]
n_samples = X_train.shape[0]

###### grid search #######

#Initialize vectors for saving values
eta_vals = np.logspace(-6, -1, 6)
lmbd_vals = np.hstack((np.array([0]), np.logspace(-6, -1, 6)))
train_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
test_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
train_R2 = np.zeros((len(eta_vals), len(lmbd_vals)))
test_R2 = np.zeros((len(eta_vals), len(lmbd_vals)))
best_train_mse = 10.
best_test_mse = 10.
#Loop through the etas and lambdas
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        #Make neural network
        ffnn = NeuralNetwork(X_train, 
                             y_train, 
                             batch_size=int(n_samples/m), 
                             n_categories = 1,
                             epochs = 100, 
                             n_hidden_neurons = 20, 
                             eta = eta,
                             lmbd = lmbd,
                             input_activation = 'sigmoid',
                             output_activation = 'linear',
                             cost_function = 'MSE')
        ffnn.add_layer(20, activation_method = 'sigmoid')
        #ffnn.add_layer(10, activation_method = 'tanh')
        
        #Train network
        ffnn.train()
        
        #Save predictions
        y_tilde_train = ffnn.predict(X_train)
        y_tilde_test = ffnn.predict(X_test)
        
        #Save metrics into exportable matrices
        train_mse[i][j], train_R2[i][j] = statistics.calc_statistics(y_train, y_tilde_train)
        test_mse[i][j], test_R2[i][j] = statistics.calc_statistics(y_test, y_tilde_test)
        
        if best_train_mse > train_mse[i][j]:
            best_train_mse = train_mse[i][j]
            best_y_tilde_train = y_tilde_train
        
        if best_test_mse > test_mse[i][j]:
            best_test_mse = test_mse[i][j]
            best_y_tilde_test = y_tilde_test
            
        
        #print metrics
        print('Learning rate: ', eta)
        print('lambda: ', lmbd)
        print('Train. mse = ', train_mse[i][j], 'R2 = ', train_R2[i][j])
        print('Test. mse = ', test_mse[i][j], 'R2 = ', test_R2[i][j])
        print('\n')
            
            
#Visualization        
show_heatmap_mse_R2(lmbd_vals, eta_vals, train_mse, test_mse, train_R2, test_R2)

#z = y_tilde_test
#x = testset.df['N']
#y = testset.df['Z']

#Rescale for plotting
rescaled_dataset = AME16.rescale_back(x = AME16.test_x_1d, y = best_y_tilde_test)
x, y, z = rescaled_dataset[:,0], rescaled_dataset[:,1], rescaled_dataset[:,-1] 


#generate plain dataset for plotting
neut = AME16.df['N'].to_numpy()
prot = AME16.df['Z'].to_numpy()
Mexc = AME16.df['MassExcess'].to_numpy()

#Plot best fit
plot_3d_terrain(x, y, z, neut, prot, Mexc)


