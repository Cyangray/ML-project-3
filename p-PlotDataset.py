import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#Load the datasets
with open('datasets.pkl', 'rb') as input:
    Datasets = pickle.load(input)
AME12 = Datasets[0]
AME16 = Datasets[1]
testset = Datasets[2]

def draw_dataset(dataset, quantity):
    
    if quantity == 'B':
        title_query = 'total binding energy'
    elif quantity == 'B/A':
        title_query = 'binding energy per nucleon'
    else:
        title_query = quantity
    
    # Magic numbers
    Mnumbers = np.array([2, 8, 20, 28, 50, 82, 126]) -3
    
    fig, ax = plt.subplots()
    sns.set()
    
    plot_df = dataset.df.pivot('Z','N', quantity)
    plot_df.dropna()
    sns.heatmap(plot_df, square = True, cmap = 'plasma', ax=ax)
    ax.hlines(Mnumbers, *ax.get_xlim())
    ax.vlines(Mnumbers, *ax.get_ylim())
    ax.invert_yaxis()
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title(dataset.ds_name + ' values for ' + title_query)
    
    plt.show()
