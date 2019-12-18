import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from visualization import plot_features
from sklearn import preprocessing
import sys
from functions import franke_function, is_odd

class dataset():
    
    def __init__(self, filename, header = 0, skiprows = 0, index_col = 0):
        nanDict = {}
        self.normalized = False
        self.resort = 0
        self.header = header
        self.skiprows = skiprows
        if filename != 0:
            self.df = pd.read_excel(filename, header = header, skiprows = skiprows, index_col = index_col, na_values=nanDict)
            self.pandas_df = True
        else:
            self.pandas_df = False
    
    def generate_franke(self, n, noise ):
        """ Generate franke-data with randomised noise in a n*n random grid. """
        self.n = n
        self.noise = noise
        self.N = n*n #Number of datapoints (in a square meshgrid)
        self.input_variables = 2
        self.x0_mesh, self.x1_mesh = np.meshgrid(np.random.uniform(0, 1, n), np.random.uniform(0, 1, n))
        self.y_mesh = franke_function(self.x0_mesh, self.x1_mesh)
        
        self.x_1d = np.zeros((self.N, self.input_variables))
        self.x_1d[:,0] = np.ravel(self.x0_mesh)
        self.x_1d[:,1] = np.ravel(self.x1_mesh)
        self.y_1d = np.ravel(self.y_mesh)
        
        if self.noise != 0:
            self.y_1d += np.random.randn(n*n) * self.noise
            
        self.values = np.column_stack((self.x_1d, self.y_1d))
    
    def polish_and_divide(self, targetcol = -1, headercols = 0):
        '''Divide the DataFrame into data and target.
        targetcol points to the column containing the targets,
        headercols indicates whether some columns should be skipped from the dataset.'''
        headerrows = self.header
        if self.pandas_df:
            self.values = np.copy(self.df.to_numpy())
            self.feature_names = np.copy(self.df.columns)
            
        self.x_1d = self.values[headercols: , headerrows : targetcol]
        self.y_1d = self.values[headercols: , targetcol]
        self.N = self.x_1d.shape[0]
        
    def normalize_dataset(self):
        ''' Uses the scikit-learn preprocessing tool for scaling the datasets. 
        Use the MinMaxScaler.'''
        self.normalized = True
        self.x_1d_unscaled = self.x_1d.copy()
        self.y_1d_unscaled = self.y_1d.copy()
        dataset_matrix = self.values
        self.scaler = preprocessing.MinMaxScaler().fit(dataset_matrix)
        transformed_matrix = self.scaler.transform(dataset_matrix)
        self.x_1d = transformed_matrix[:,:-1]
        self.y_1d = transformed_matrix[:,-1]
        
    def rescale_back(self, x=0, y=0, split = False):
        """ After processing, the data must be scaled back to normal by scalers 
        inverse_transform for mainly plotting and validating purposes. If no x and y
        are given, the scaled back version of the training set (if train-test
        splitted) is returned.
        The split argument is boolean, if True it returns the rescaled input and
        output/target values separately. This is useful when for example one has
        to only rescale a prediction."""
        #self.normalized = False
        if isinstance(x, int):
            x = self.x_1d
        if isinstance(y, int):
            y = self.y_1d
        dataset_matrix = np.column_stack((x, y))
        rescaled_matrix = self.scaler.inverse_transform(dataset_matrix)
        if split:
            x_out = rescaled_matrix[:,:-1]
            y_out = rescaled_matrix[:,-1]
            return x_out, y_out
        else:
            return rescaled_matrix
    
    def sort_train_test(self, ratio=0.2, random=True):
        '''sorts the dataset into a training and a test set. Ratio is a number
        between 0 and 1, giving the ratio of the test set'''
        N_test = int(self.N*ratio)
        self.training_indices = []
        self.test_indices = []
        
        if random:
            '''Loop all indexes, Generate a random number, see if it lies below
            a treshold given by the ratio. if so, this will end up in the training set'''
            idx = 0
            for idx in range(self.N):
                random_number = np.random.rand()
                if random_number < ratio:
                    self.test_indices.append(idx)
                else:
                    self.training_indices.append(idx)
        else:
            '''shuffles randomly and splits into train and test.'''
            split = np.arange(self.N)
            np.random.shuffle(split)
            self.training_indices = split[N_test:]
            self.test_indices = split[:N_test]
            
        self.fill_array_test_training()
            
        
    def sort_in_k_batches(self, k, random=True, minibatches = False):
        """ Sorts the data into k batches, i.e. prepares the data for k-fold cross
        validation. Recommended numbers are k = 3, 4 or 5. "random" sorts the
        dataset randomly. if random==False, it sorts them statistically"""
            
        idx = 0
        N = self.N
        batches_idxs = [[] for i in range(k)]
        limits = [i/k for i in range(k+1)]
        
        if random:
            '''Loop all indexes, Generate a random number, see where it lies in k 
            evenly spaced intervals, use that to determine in which set to put
            each index'''
            while idx < N:
                random_number = np.random.rand()
                for i in range(k):
                    if limits[i] <= random_number < limits[i+1]:
                        batches_idxs[i].append(idx)
                        break
                idx += 1
            
        else: 
            '''Statistical sorting lists int values, shuffles randomly and splits into k pieces.'''
            split = np.arange(N)
            np.random.shuffle(split)
            #exp_limits = [elem * N for elem in limits] 
            limits = [int(elem*N) for elem in limits]
            for i in range(k):
                batches_idxs[i] = split[limits[i] : limits[i+1]]
                
        if minibatches:
            self.m = k
            self.m_idxs = batches_idxs
        else:
            self.k = k
            self.k_idxs = batches_idxs
                
    def sort_training_test_kfold(self, i):
        """After sorting the dataset into k batches, pick one of them and this one 
        will play the part of the test set, while the rest will end up being 
        the training set. the input i should be an integer between 0 and k-1, and it
        picks the test set. """
        self.test_indices = self.k_idxs[i]
        self.training_indices = []
        for idx in range(self.k):
            if idx != i:
                self.training_indices += self.k_idxs[idx]


    def fill_array_test_training(self):
        """ Fill the arrays, eg. test_x_1d and x_1d for x, y and z with
        the actual training data according to how the indices were sorted in 
        sort_training_test_kfold."""
        testing = self.test_indices ; training = self.training_indices

        self.reload_data()

        self.test_x_1d = self.x_1d[testing, :]
        self.test_y_1d = self.y_1d[testing]
        
        self.x_1d = self.x_1d[training, :]
        self.y_1d = self.y_1d[training]
        
        # Redefine lengths for training and testing.
        self.N = len(training)
        self.N_testing = len(testing)
        
        
    def reload_data(self):
        """ Neat system for automatically make a backup of data sets if you resort. """
        if self.resort < 1:
            np.savez("backup_data", N=self.N, x=self.x_1d, y=self.y_1d)
        else: # self.resort >= 1:
            data = np.load("backup_data.npz")
            self.N = data["N"]
            self.x_1d = data["x"]
            self.y_1d = data["y"]
        self.resort = 10
        
        
        
class credit_card_dataset(dataset):
    '''Child class of dataset, giving more methods, specific for the credit card
    dataset, with some hard-coded, dataset-specific values to give a neater experience in interface'''
    def __init__(self, filename):
        super().__init__(filename, header = 0, skiprows = 1, index_col = 0)
    
    def CreditCardPolish(self, drop0 = True):
        self.df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)
        if drop0:
            self.df = self.df.drop(self.df[(self.df.BILL_AMT1 == 0) &
                    (self.df.BILL_AMT2 == 0) &
                    (self.df.BILL_AMT3 == 0) &
                    (self.df.BILL_AMT4 == 0) &
                    (self.df.BILL_AMT5 == 0) &
                    (self.df.BILL_AMT6 == 0)].index)
    
            self.df = self.df.drop(self.df[(self.df.PAY_AMT1 == 0) &
                    (self.df.PAY_AMT2 == 0) &
                    (self.df.PAY_AMT3 == 0) &
                    (self.df.PAY_AMT4 == 0) &
                    (self.df.PAY_AMT5 == 0) &
                    (self.df.PAY_AMT6 == 0)].index)
        
        super().polish_and_divide()
        
    def plot_setup(self):
        self.contbins = self.df['AGE'].max() - self.df['AGE'].min()
        self.continuous_features_idxs = [0,4,11,12,13,14,15,16,17,18,19,20,21,22]
        self.cont_rows = 7
        self.cont_cols = 2
        self.discrete_features_idxs = [1,2,3,5,6,7,8,9,10]
        self.disc_rows = 3
        self.disc_cols = 3
        self.output_labels = ["Non-default", "Default"]

class AtomicMasses(dataset):
    def __init__(self, filenames, usealldata = True):
        super().__init__(0)
        
        self.pandas_df = True
        self.usealldata = usealldata
        # Read the experimental data with Pandas
        if usealldata:
            dataset = pd.read_fwf(filenames[0], usecols=(2, 3, 4, 6, 11, 15),
                          names=('N', 'Z', 'A', 'Element', 'B/A', 'BetaDecayEnergy'),
                          widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),
                          header=39,
                          index_col=False)
            
            new_dataset_1 = pd.read_fwf(filenames[1], usecols=(6,8),
                          names=('S2n', 'S2p'),
                          widths=(1,3,1,3,3,1,10,8,10,8,10,8,10,8,10,8,10,8),
                          header=39,
                          index_col=False)
            
            new_dataset_2 = pd.read_fwf(filenames[2], usecols=(6,8),
                          names=('Sn', 'Sp'),
                          widths=(1,3,1,3,3,1,10,8,10,8,10,8,10,8,10,8,10,8),
                          header=39,
                          index_col=False)
            self.df = pd.concat([dataset, new_dataset_1, new_dataset_2], axis = 1)
        else:
            dataset = pd.read_fwf(filenames[0], usecols=(2, 3, 4, 6, 11),
                          names=('N', 'Z', 'A', 'Element', 'B/A'),
                          widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),
                          header=39,
                          index_col=False)
            self.df = dataset
                
        givenames = True
        if givenames == True:
            for idx, row in self.df.iterrows():
                self.df.rename(index={idx : (str(self.df.loc[idx, 'A']) + self.df.loc[idx, 'Element']) }, inplace=True)
        self.df.pop('Element')
        
    def add_oddeven(self):
        '''add a column to the dataset telling if the nucleus is even-even, odd-even
        or odd-odd. '''
        self.df['Parity'] = 0
        for idx, row in self.df.iterrows():
            if is_odd(row['N']) or is_odd(row['Z']):
                if is_odd(row['N']) and is_odd(row['Z']): #odd-odd
                    self.df.at[idx, 'Parity'] = 0
                else:               #odd-even
                    self.df.at[idx, 'Parity'] = 1
            else: #even-even
                self.df.at[idx, 'Parity'] = 2
            
    
    def AMPolishDivide(self):
        # Extrapolated values are indicated by '#' in place of the decimal place, so
        # the Ebinding column won't be numeric. Coerce to float and drop these entries.
        # Do the same for the BetaDecayEnergy, S2n, S2p, Sn and Sp columns where there are no results,
        # meaning where the entry is '*'
        
        #72Cu, 209Ac is found in AME12, but only experimental in AME16. Value is dropped
        AME12butnot16 = ['72Cu', '209Ac', '70Co', '80Zr', '209Th']
        for nucleus in AME12butnot16:
            self.df.drop(nucleus, inplace = True)
        
        
        if self.usealldata:
            columns = ['B/A', 'BetaDecayEnergy', 'S2n', 'S2p', 'Sn', 'Sp']
        else:
            columns = ['B/A']
        for col in columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df = self.df.dropna()
            # Convert from keV to MeV.
            self.df[col] /= 1000
        
        target = self.df.pop('B/A')
        self.df['B/A'] = target
        self.values = np.copy(self.df.to_numpy())
        self.feature_names = np.copy(self.df.columns)
        
        
        self.x_1d = self.values[:, :-1]
        self.y_1d = self.values[:,-1]
        self.N = self.x_1d.shape[0]
        
    def sort_train_test(self, AME12, useAME12 = False):
        '''Method overwriting the parent class' sort_train_test, taking as input
        the previously gotten labels for the train and test nuclei.
        With useEMA12 = True, the values from the EMA12 database will be used as
        training set, otherwise the updated values from EMA16 will be used. Default
        to False'''
        
        labels_test = np.setdiff1d(self.df.index.to_numpy(), AME12.df.index.to_numpy())
        AME16labels = self.df.index.to_numpy()
        test_idxs = []
        for i, label in enumerate(labels_test):
            test_idxs.append(np.where(AME16labels == label)[0][0])
        
        self.test_x_1d = self.x_1d[np.array(test_idxs, dtype=int)[:], :]
        self.test_y_1d = self.y_1d[np.array(test_idxs, dtype=int)[:]]
        
        labels_train = AME12.df.index.to_numpy()
        train_idxs = []
        if useAME12:
            trainvalues_x = AME12.x_1d
            trainvalues_y = AME12.y_1d
        else:
            for i, label in enumerate(labels_train):
                train_idxs.append(np.where(AME16labels == label)[0][0])
            trainvalues_x = self.x_1d[np.array(train_idxs, dtype=int)[:], :]
            trainvalues_y = self.y_1d[np.array(train_idxs, dtype=int)[:]]
        
        self.x_1d = trainvalues_x
        self.y_1d = trainvalues_y
        
        # Redefine lengths for training and testing.
        self.N = self.x_1d.shape[0]
        self.N_testing = self.x_1d.shape[0]
        
        
    