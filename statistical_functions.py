import numpy as np
from sklearn.metrics import r2_score, roc_auc_score
import scikitplot as skplt
from functions import make_onehot

def calc_MSE(y, y_tilde):
    mse = 0
    n=len(y)
    for i in range(n):
        mse += (y[i] - y_tilde[i])**2
    return mse/n


def calc_R2_score(y, y_tilde):
    mse = 0
    ms_avg = 0
    n=len(y)
    mean_y = np.mean(y)
    for i in range(n):
        mse += (y[i] - y_tilde[i])**2
        ms_avg += (y[i] - mean_y)**2
    return 1. - mse/ms_avg

def calc_R2_score_sklearn(y, y_tilde):
    return r2_score(y, y_tilde)


def calc_statistics(y, y_tilde):
    mse = calc_MSE(y, y_tilde)
    calc_r2 = calc_R2_score(y, y_tilde)
    return mse, calc_r2

def calc_bias_variance(y, y_tilde):
    """ Calculate the bias and the variance of a given model"""
    n = len(y)
    Eytilde = np.mean(y_tilde)
    bias = 1/n * np.sum((y - Eytilde)**2)
    variance = 1/n * np.sum((y_tilde - Eytilde)**2)
    return bias, variance

def calc_cumulative_auc(target, pred):
    '''calculate the area under the cumulative chart'''
    x_data, y_data = skplt.helpers.cumulative_gain_curve(target, pred[:,1])
    area = np.trapz(y_data, x = x_data)
    return area
    
def calc_area_ratio(target,pred):
    max_area = calc_cumulative_auc(target, make_onehot(target))
    auc = calc_cumulative_auc(target, pred)
    return (auc - 0.5)/(max_area - 0.5)

def print_mse(mse):
    print("Average mse: ", np.average(mse))
    print("Best mse: ", np.min(mse[np.argmin(np.abs(np.array(mse)))]))

def print_R2(R2):
    print("Average R2: ", np.average(R2))
    print("Best R2: ", R2[np.argmax(np.array(R2))])
    
def calc_accuracy(target, y_tilde, threshold = 0.5):
    '''Take as input the modeled y_tilde and the correct values target, returns the
    accuracy of the model'''
        
    y10 = np.zeros(np.shape(y_tilde))
    right_guesses = 0
    
    for i, yi in enumerate(y_tilde):
        if yi > threshold:
            y10[i] = 1
        if y10[i] == target[i]:
            right_guesses += 1
            
    accy = right_guesses / len(target)
    return accy

def calc_rocauc(target, y_tilde):
    return roc_auc_score(target, y_tilde)


