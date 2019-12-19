import numpy as np
from fit_matrix import fit
import statistical_functions as statistics
import pickle
from visualization import show_heatmap_mse_R2, plot_3d_terrain

#'Ridge' or 'LASSO'?
isridge = False
if isridge:
    method = 'Ridge'
else:
    method = 'LASSO'
    
#Polynomial degrees and L2 values to loop through in the grid search
lmbd_vals = np.hstack((np.array([0]), np.logspace(-6, -1, 6)))
deg_vals = np.linspace(0, 15, 16, dtype = int)

#Stochastic gradient descent parameters
Niterations = 10e5 #also valid as maxiter for LASSO

#Random seed
np.random.seed(1234)

# Load datasets
with open('datasets.pkl', 'rb') as input:
    Datasets = pickle.load(input)
AME12 = Datasets[0]
AME16 = Datasets[1]
testset = Datasets[2]

#Divide in train and test
AME16.sort_train_test(AME12, useAME12 = False)

#Make model
AME16Fit = fit(AME16)

#Grid search
train_mse = np.zeros((len(deg_vals), len(lmbd_vals)))
test_mse = np.zeros((len(deg_vals), len(lmbd_vals)))
train_R2 = np.zeros((len(deg_vals), len(lmbd_vals)))
test_R2 = np.zeros((len(deg_vals), len(lmbd_vals)))
best_train_mse = 10.
best_test_mse = 10.
best_OLS_mse = 10.
for i, deg in enumerate(deg_vals):
    for j, lmbd in enumerate(lmbd_vals):
        #Create polynomial design matrix for train and test sets
        X_train = AME16Fit.create_design_matrix(deg = deg)
        
        #collect information about training set
        if lmbd == 0.0:
            y_tilde_train, betas = AME16Fit.fit_design_matrix_numpy()
        elif method == 'Ridge':
            y_tilde_train, betas = AME16Fit.fit_design_matrix_ridge(lambd = lmbd)
        elif method == 'LASSO':
            y_tilde_train, betas = AME16Fit.fit_design_matrix_lasso(lambd = lmbd, maxiter = Niterations)
        y_train = AME16.y_1d
        
        #collect information about test set
        X_test = AME16Fit.create_design_matrix(x = AME16.test_x_1d, deg = deg)
        y_tilde_test = AME16Fit.test_design_matrix(betas, X = X_test)
        y_test = AME16.test_y_1d
        
        #Save metrics into exportable matrices
        train_mse[i][j], train_R2[i][j] = statistics.calc_statistics(y_train, y_tilde_train)
        test_mse[i][j], test_R2[i][j] = statistics.calc_statistics(y_test, y_tilde_test)
        
        #OLS will (almost) always score better. Divide for better statistics
        if lmbd == 0.0:
            if best_OLS_mse > test_mse[i][j]:
                best_OLS_mse = test_mse[i][j]
                best_OLS_y_tilde_test = y_tilde_test
                best_OLS_deg = deg
        else:
            if best_test_mse > test_mse[i][j]:
                best_test_mse = test_mse[i][j]
                best_y_tilde_test = y_tilde_test
                best_deg = deg
                best_lambda = lmbd
        
        #print progress
        print('Degree: ', deg, 'lambda: ', lmbd)
        
        
#Visualization
show_heatmap_mse_R2(lmbd_vals, deg_vals, train_mse, test_mse, train_R2, test_R2, method = 'Regr')

#Rescale for plotting
rescaled_dataset = AME16.rescale_back(x = AME16.test_x_1d, y = best_y_tilde_test)
_, y_OLS = AME16.rescale_back(x = AME16.test_x_1d, y = best_OLS_y_tilde_test, split = True)
x, y, z = rescaled_dataset[:,0], rescaled_dataset[:,1], rescaled_dataset[:,-1] 

#generate plain dataset for plotting
neut = AME16.df['N'].to_numpy()
prot = AME16.df['Z'].to_numpy()
Mexc = AME16.df['B'].to_numpy()

#Plot best fit
plot_3d_terrain(x, y, z, neut, prot, Mexc)

#Save predictions to file
Regr_data = np.concatenate((x[:,np.newaxis],y[:,np.newaxis],z[:,np.newaxis]), axis = 1)
Regr_OLS = np.concatenate((x[:,np.newaxis],y[:,np.newaxis],y_OLS[:,np.newaxis]), axis = 1)
header = 'Regression, OLS. (Best) degree: ' + str(best_OLS_deg) + '. Columns: N, Z, B.'
np.savetxt('Regression_OLS_data.txt', Regr_OLS, header = header)
if isridge:
    header = 'Regression, Ridge. (Best) degree: ' + str(best_deg) + '. Associated L2 hyperparameter: ' + str(best_lambda) + '. Columns: N, Z, B.'
    np.savetxt('Regression_ridge_data.txt', Regr_data, header = header)
else:
    header = 'FFNN, LASSO. (Best) degree: ' + str(best_deg) + '. Associated L2 hyperparameter: ' + str(best_lambda) + '. Columns: N, Z, B.'    
    np.savetxt('Regression_LASSO_data.txt', Regr_data, header = header)
        