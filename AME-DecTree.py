import numpy as np
from visualization import show_heatmap_mse_R2, plot_3d_terrain
from fit_matrix import fit
import statistical_functions as statistics
from sklearn.tree import DecisionTreeRegressor
import pickle
import xgboost as xgb

#Random seed
np.random.seed(1234)

#Normal decision trees with pruning, or XGBoost?
XGBoost = False

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
depth_vals = np.linspace(1, 10, 10)
lmbd_vals = np.hstack((np.array([0.0]), np.logspace(-6, -1, 6)))
train_mse = np.zeros((len(depth_vals), len(lmbd_vals)))
test_mse = np.zeros((len(depth_vals), len(lmbd_vals)))
train_R2 = np.zeros((len(depth_vals), len(lmbd_vals)))
test_R2 = np.zeros((len(depth_vals), len(lmbd_vals)))
best_test_mse = 10.
#Loop through the etas and lambdas
for i, depth in enumerate(depth_vals):
    for j, lmbd in enumerate(lmbd_vals):
        #Make decision tree
        if XGBoost:
            dectree = xgb.XGBRegressor(max_depth = int(depth), reg_lambda = lmbd)
        else:
            dectree = DecisionTreeRegressor(max_depth = depth, ccp_alpha = lmbd)
            
        # Fit decision tree
        dectree.fit(X_train, y_train)
        
        #Save predictions
        y_tilde_train = dectree.predict(X_train)
        y_tilde_test = dectree.predict(X_test)
        
        #Save metrics into exportable matrices
        train_mse[i][j], train_R2[i][j] = statistics.calc_statistics(y_train, y_tilde_train)
        test_mse[i][j], test_R2[i][j] = statistics.calc_statistics(y_test, y_tilde_test)
        
        #Save best model
        if best_test_mse > test_mse[i][j]:
            bestdepth = depth
            bestlambda = lmbd
            best_test_mse = test_mse[i][j]
            best_y_tilde_test = y_tilde_test
            
        #print progress
        print('Depth: ', depth, 'lambda: ', lmbd)
            
#Visualization        
show_heatmap_mse_R2(lmbd_vals, depth_vals, train_mse, test_mse, train_R2, test_R2, method = 'Trees')

#Rescale for plotting and statistics
rescaled_dataset = AME16.rescale_back(x = AME16.test_x_1d, y = best_y_tilde_test)
x, y, z = rescaled_dataset[:,0], rescaled_dataset[:,1], rescaled_dataset[:,-1] 

#generate plain dataset for plotting and statistics
neut = AME16.df['N'].to_numpy()
prot = AME16.df['Z'].to_numpy()
Mexc = AME16.df['B'].to_numpy()

#Plot best fit
plot_3d_terrain(x, y, z, neut, prot, Mexc)

#Save best fit predictions
DecisionTree_data = np.concatenate((x[:,np.newaxis],y[:,np.newaxis],z[:,np.newaxis]), axis = 1)
if XGBoost:
    header = 'XGBoost method. (Best) depth: ' + str(int(bestdepth)) + '. Associated L2 hyperparameter: ' + str(bestlambda) + '. Columns: N, Z, ExcessMass.'
    np.savetxt('XGBoost_data.txt', DecisionTree_data, header = header)
else:
    header = 'Decision tree method, with pruning. (Best) depth: ' + str(int(bestdepth)) + '. Associated L2 hyperparameter: ' + str(bestlambda) + '. Columns: N, Z, ExcessMass.'    
    np.savetxt('DecisionTree_data.txt', DecisionTree_data, header = header)