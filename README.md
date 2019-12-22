The report with a description of most of the files is found in file Report_project_3.pdf

The proper code files in the repository are make_datasets.py, AME-Regression.py, AME-FFNN.py, AME-DecTrees.py and comparisons.py
files p-OldNewNuclei.py and p-PlotDataset.py draw some of the plots in the report.
All files ending in .txt are outputs of the different functions, where all those whose name finished with _data.txt are the outputs of the different regression scripts.
File dataset.pkl is the output of make_datasets.py, which will then be imported by all other functions through the pickle python package.
All other .py files are either classes or function libraries used in the regression tasks.

In the folder published_data are collected the values found by Utama et al. which I try to reproduce and compare my results with.
In the folder data are collected the Atomic Mass Evaluation datasets for 2012 and 2016 used for the modelings.

All codes as-is reproduce the results given in the report, as a random seed is given at the start of every code involving pseudo-random calculations.
