Repository for the data, reanalyses, and computational modeling presented in:

Popov, V. & Reder, L. (2018). Frequency Effects on Memory: A Resource-Limited Theory. PsyArXiv. http://doi.org/10.17605/OSF.IO/DSX6Y. 


1. All behavioral analyses stored in 'analyses/' are run with R within RStudio. They run with paths relative to the main parent folder. Most straightforward way to run them is to open the 123-prior-item-effects.Rproj file, which should set the correct working directory.
2. All behavioral analyses operate on the .csv data files stored in 'data/'
3. The 'sac/' folder contains the main Python code (Python 3) for the SAC infrastructure. It is loaded as a python package when building individual models for each study
4. The individual models for each study are under 'models/'
5. The sac models load the behavioral data stored in 'data/formatted_for_modeling', which contains the same dataset as 'data/', but with some additions necessary for performing the model fitting procedures.