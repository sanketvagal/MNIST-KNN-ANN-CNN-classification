# MNIST-KNN-ANN-CNN-classification



Implemented K-Fold cross validation (K=5). Built CNN with the basic layer architecture, a custom CNN for predicting on images with unknown labels, and K-NN classifiers with (K=1, K=5 and K=10). For the procured dataset, ANN was used instead of CNN, as it is not an image based dataset. Performed classification on the following datasets: 

1. MNIST dataset 

2. Diabetic Retinopathy Debrecen Data Set. (2014). UCI Machine Learning Repository.   
    Source: https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set

Calculated AUC values for the procured dataset. 

NOTE: for a fair comparison, K-Fold randomization was only performed once, with any selected samples for training applied to the creation of all classifier types (SVM, RF, KNN) in an identical manner (i.e. the exact same set of training data used to construct each model being compared to ensure a fair comparison).

Calculated K Fold validated error rate for each of the classifiers.
