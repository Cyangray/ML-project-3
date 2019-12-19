import numpy as np
from visualization import show_heatmap_mse_R2, plot_3d_terrain
from neural_network import NeuralNetwork
from fit_matrix import fit
import statistical_functions as statistics
import pickle

#Random seed
np.random.seed(1234)

#Stochastic gradient descent parameters
m = 20           #Number of minibatches
Niterations = 5000

#Activation function
acts = ['sigmoid', 'tanh']

#output activation function
out_acts = ['linear', 'tanh']

for act in acts:
    for out_act in out_acts:

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
        X_train = AME16Fit.create_design_matrix(deg = 0)
        X_test = AME16Fit.create_design_matrix(x = AME16.test_x_1d, deg = 0)
        
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
                                     input_activation = act,
                                     output_activation = out_act,
                                     cost_function = 'MSE')
                ffnn.add_layer(20, activation_method = act)
                ffnn.add_layer(10, activation_method = act)
                
                #Train network
                ffnn.train()
                
                #Save predictions
                y_tilde_train = ffnn.predict(X_train)
                y_tilde_test = ffnn.predict(X_test)
                
                #Save metrics into exportable matrices
                train_mse[i][j], train_R2[i][j] = statistics.calc_statistics(y_train, y_tilde_train)
                test_mse[i][j], test_R2[i][j] = statistics.calc_statistics(y_test, y_tilde_test)
                
                if best_test_mse > test_mse[i][j]:
                    best_test_mse = test_mse[i][j]
                    best_y_tilde_test = y_tilde_test
                    best_eta = eta
                    best_lambda = lmbd
                    
                #print progress
                print('Learning rate: ', eta, 'lambda: ', lmbd)
                    
        #Visualization        
        show_heatmap_mse_R2(lmbd_vals, eta_vals, train_mse, test_mse, train_R2, test_R2, method = 'NN')
        
        #Rescale for plotting and stats
        rescaled_dataset = AME16.rescale_back(x = AME16.test_x_1d, y = best_y_tilde_test)
        x, y, z = rescaled_dataset[:,0], rescaled_dataset[:,1], rescaled_dataset[:,-1] 
        
        #generate plain dataset for plotting
        neut = AME16.df['N'].to_numpy()
        prot = AME16.df['Z'].to_numpy()
        Mexc = AME16.df['B'].to_numpy()
        
        #Plot best fit
        plot_3d_terrain(x, y, z, neut, prot, Mexc)
        
        #Save Predictions
        FFNN_data = np.concatenate((x[:,np.newaxis],y[:,np.newaxis],z[:,np.newaxis]), axis = 1)
        if out_act == 'linear':
            if act == 'sigmoid':
                header = 'FFNN, sigmoid activation, linear out. (Best) eta: ' + str(best_eta) + '. Associated L2 hyperparameter: ' + str(best_lambda) + '. Columns: N, Z, B.'
                np.savetxt('FFNN_sigmoid_lin_data.txt', FFNN_data, header = header)
            else:
                header = 'FFNN, tanh activation, linear out. (Best) eta: ' + str(best_eta) + '. Associated L2 hyperparameter: ' + str(best_lambda) + '. Columns: N, Z, B.'    
                np.savetxt('FFNN_tanh_lin_data.txt', FFNN_data, header = header)
        else:
            if act == 'sigmoid':
                header = 'FFNN, sigmoid activation, tanh out. (Best) eta: ' + str(best_eta) + '. Associated L2 hyperparameter: ' + str(best_lambda) + '. Columns: N, Z, B.'
                np.savetxt('FFNN_sigmoid_tanh_data.txt', FFNN_data, header = header)
            else:
                header = 'FFNN, tanh activation, tanh out. (Best) eta: ' + str(best_eta) + '. Associated L2 hyperparameter: ' + str(best_lambda) + '. Columns: N, Z, B.'    
                np.savetxt('FFNN_tanh_tanh_data.txt', FFNN_data, header = header)
