import numpy as np
from dataset_objects import AtomicMasses
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pickle

#Load the datasets
with open('datasets.pkl', 'rb') as input:
    Datasets = pickle.load(input)

AME12 = Datasets[0]
AME16 = Datasets[1]
AMEnew = Datasets[2]

#A plot
Mnumbers = np.array([2, 8, 20, 28, 50, 82, 126]) -3

plot_df = AMEnew.df.pivot('Z','N','S2p')
ax = sns.heatmap(plot_df, square = True, cmap = 'plasma')
ax.hlines(Mnumbers, *ax.get_xlim())
ax.vlines(Mnumbers, *ax.get_ylim())
ax.invert_yaxis()
