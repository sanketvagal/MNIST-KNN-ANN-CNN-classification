# MNIST-KNN-ANN-CNN-classification



Implemented K-Fold cross validation (K=5). Built CNN with the basic layer architecture, a custom CNN for predicting on images with unknown labels, and K-NN classifiers with (K=1, K=5 and K=10). For the procured dataset, ANN was used instead of CNN, as it is not an image based dataset. Performed classification on the following datasets: 

1. MNIST dataset 

2. Diabetic Retinopathy Debrecen Data Set. (2014). UCI Machine Learning Repository.   
    Source: https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set

Calculated AUC values for the procured dataset. 

NOTE: for a fair comparison, K-Fold randomization was only performed once, with any selected samples for training applied to the creation of all classifier types (SVM, RF, KNN) in an identical manner (i.e. the exact same set of training data used to construct each model being compared to ensure a fair comparison).

Calculated K Fold validated error rate for each of the classifiers.

## Disclaimer

The following academic assignment is uploaded on GitHub solely for educational purposes and to showcase my work. The content contained within is the original work of the author, and any sources used have been properly cited and credited. This assignment is not intended to be used for any other purpose, including but not limited to commercial gain or academic dishonesty. The author assumes no responsibility or liability for any misuse or misinterpretation of the information contained within this assignment. The reader is solely responsible for verifying the accuracy and applicability of the information contained herein.

If the professor or any relevant authority has concerns regarding this repository, please do not hesitate to contact me directly. I am open to feedback and willing to make any necessary revisions to ensure that this work aligns with academic integrity standards.
