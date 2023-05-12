from pathlib import Path
from typing import Dict, Literal, Tuple, Union

import numpy as np
import pandas as pd
from keras import Input, Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, ReLU
from scipy.io import loadmat
from scipy.io.arff import loadarff
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow import keras


def process_mnist_train() -> Tuple[np.ndarray, np.ndarray]:
    """Load the MNIST dataset in features and labels

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and labels data
    """
    # Loading the dataset
    numbers_dataset = loadmat(Path(__file__).resolve().parent / "NumberRecognitionBiggest.mat")

    # Setting up samples data and labels
    X = numbers_dataset["X_train"]
    y = numbers_dataset["y_train"].squeeze()
    return X, y


def process_mnist_test() -> np.ndarray:
    """Load the MNIST test dataset in features

    Returns:
        np.ndarray: Features data
    """
    # Loading the dataset
    numbers_dataset = loadmat(Path(__file__).resolve().parent / "NumberRecognitionBiggest.mat")

    # Setting up samples data and labels
    X_test: np.ndarray = numbers_dataset["X_test"]
    return X_test


def cnn_q1_model() -> Sequential:
    """Generate the Keras CNN model for question 1, based on keras_mnist.py demo file on moodle

    Returns:
        Sequential: Compiled Keras Sequential CNN model
    """
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(1, kernel_size=3, padding="same", use_bias=True),
            ReLU(),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def cnn_best_model() -> Sequential:
    """Generate the best Keras CNN model for question 4

    Returns:
        Sequential: Compiled Keras Sequential CNN model
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=5, activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size=5, activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.summary()

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def preprocess_cnn(
    X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, train: int, test: int, predict: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess data to be used for Keras CNN models for every fold

    Args:
        X (Union[pd.DataFrame, np.ndarray]): Features data
        y (np.ndarray): Labels data
        train (int): kfold train index
        test (int): kfold test index
        predict (bool): Preprocessing based on which model to use

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Preprocessed training and testing features and labels
                                                                data for the current fold
    """
    # Keras by default does not scale MNIST, so we need to scale and convert to float
    # The MNIST data also doesn't include channel information, so we add it with np.expand_dims
    X_train = X[train].astype("float32") / 255
    X_train = np.expand_dims(X_train, -1)

    X_test = X[test].astype("float32") / 255
    X_test = np.expand_dims(X_test, -1)

    # One hot encoding based on the loss function used
    if predict:
        y_train = y[train]
        y_test = y[test]
    else:
        y_train = keras.utils.to_categorical(y[train], 10)
        y_test = keras.utils.to_categorical(y[test], 10)

    return X_train, y_train, X_test, y_test


def preprocess_knn(
    X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], train: int, test: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess data to be used for scikit learn KNN and ANN models for every fold

    Args:
        X (Union[np.ndarray, pd.DataFrame]): Features data
        y (Union[np.ndarray, pd.Series]): Labels data
        train (int): kfold train index
        test (int): kfold test index

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Preprocessed training and testing features and labels
                                                                data for the current fold
    """

    # Performing preprocessing based on the type of data
    if isinstance(X, np.ndarray):
        X_train = X[train].reshape(X[train].shape[0], X[train].shape[1] * X[train].shape[2])
        X_test = X[test].reshape(X[test].shape[0], X[test].shape[1] * X[test].shape[2])
    else:
        X_train = X.iloc[train]
        X_test = X.iloc[test]

    y_train = y[train]
    y_test = y[test]

    return X_train, y_train, X_test, y_test


def calculate_kfold_scores(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    clf_index: list,
    classifiers: list,
    predict: bool,
) -> pd.DataFrame:
    """Calculate error rates for each fold using k-fold cross validation, and the mean of all the error rates.

    Args:
        X (ArrayLike): Features data
        y (ArrayLike): Labels data
        clf_index (list): List of names of clasifiers
        classifiers (list): Models of classifiers
        predict (bool): Bool to decide whether to perform prediction on test data

    Returns:
        pd.DataFrame: k-fold error rates and mean of all error rates for each classifier.
    """

    # Creating a StratifiedKFold object with n_splits, shuffle, and random_state set to fixed seed
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    # Creating index and columns for scores dataframe
    fold_cols = ["fold1", "fold2", "fold3", "fold4", "fold5"]

    # Creating a dataframe with above defined index and columns
    kfold_scores = pd.DataFrame(index=clf_index, columns=fold_cols)
    kfold_scores.index.name = "classifier"

    # Perform k-fold cross validation over the classifiers
    for name, clf in zip(clf_index, classifiers):
        k_errors = []
        for train, test in kfold.split(X, y):

            # Preprocess data based on the classifier being used
            if "cnn" in name:
                if predict:
                    X_train, y_train, X_test, y_test = preprocess_cnn(X, y, train, test, predict)
                    clf.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1)
                else:
                    X_train, y_train, X_test, y_test = preprocess_cnn(X, y, train, test, predict)
                    clf.fit(X_train, y_train, batch_size=8, epochs=1, validation_split=0.1)
                score = clf.evaluate(X_test, y_test, verbose=1)  # test

                k_errors.append(1 - score[1])

            elif "knn" or "ann" in name:
                X_train, y_train, X_test, y_test = preprocess_knn(X, y, train, test)

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                k_errors.append(1 - accuracy_score(y_test, y_pred))

        kfold_scores.loc[name] = k_errors

    # Calculate mean of all fold errors for each classifier
    kfold_scores["mean"] = kfold_scores.mean(axis=1)

    # Print the k fold scores and mean of all classifiers in markdown format
    print(kfold_scores.apply(pd.to_numeric).round(3).to_markdown())

    # Perform the prediction on the test data
    if predict:
        X_test = process_mnist_test()
        y_pred = clf.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        np.save(
            Path(__file__).resolve().parent / "predictions.npy",
            y_pred.astype(np.uint8),
            allow_pickle=False,
            fix_imports=False,
        )

    return kfold_scores


def save_kfold(kfold_scores: pd.DataFrame, question: Literal[1, 3, 4]) -> None:
    from pathlib import Path

    from pandas import DataFrame

    COLS = [*[f"fold{i}" for i in range(1, 6)], "mean"]
    INDEX = {
        1: ["cnn", "knn1", "knn5", "knn10"],
        3: ["knn1", "knn5", "knn10"],
        4: ["cnn"],
    }[question]
    outname = {
        1: "kfold_mnist.json",
        3: "kfold_data.json",
        4: "kfold_cnn.json",
    }[question]
    outfile = Path(__file__).resolve().parent / outname

    # name checks
    df = kfold_scores
    if not isinstance(df, DataFrame):
        raise ValueError("Argument `kfold_scores` to `save` must be a pandas DataFrame.")
    if kfold_scores.shape[1] != 6:
        raise ValueError("DataFrame must have 6 columns.")
    if df.columns.to_list() != COLS:
        raise ValueError(
            f"Columns are incorrectly named and/or incorrectly sorted. Got:\n{df.columns.to_list()}\n"
            f"but expected:\n{COLS}"
        )
    if df.index.name.lower() != "classifier":
        raise ValueError(
            "Index is incorrectly named. Create a proper index using `pd.Series` or "
            "`pd.Index` and use the `name='classifier'` argument."
        )
    idx_items = sorted(df.index.to_list())
    for idx in INDEX:
        if idx not in idx_items:
            raise ValueError(f"You are missing a row with index value {idx}")

    if question == 3:
        anns = df.filter(regex="ann", axis=0)
        if len(anns) < 2:
            raise ValueError(
                "You are supposed to experiment with different ANN configurations, "
                'but we found less than two rows with "ann" as the index name.'
            )

    if question == 3:
        df.to_json(outfile)
        print(f"K-Fold error rates for data successfully saved to {outfile}")
        return

    # value range checks
    if question == 1:
        if df.loc["cnn", "mean"] < 0.05:
            raise ValueError(
                "Your CNN error rate is too low. Make sure you implement the CNN as provided in "
                "the assignment or example code."
            )
        if df.loc[["knn1", "knn5"], "mean"].min() > 0.04:
            raise ValueError(
                "One of your KNN-1 or KNN-5 error rates is too high. There " "is likely an error in your code."
            )
        if df.loc["knn10", "mean"] > 0.047:
            raise ValueError("Your KNN-10 error rate is too high. There is likely an error in your code.")
        df.to_json(outfile)
        print(f"K-Fold error rates for MNIST data successfully saved to {outfile}")
        return

    # must be question 4
    df.to_json(outfile)
    print(f"K-Fold error rates for custom CNN on MNIST data successfully saved to {outfile}")


def process_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Perform preprocessing on the dataset and split it in features and labels

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and labels data
    """
    # Reading the dataset (ARFF format)
    drd_dataset = loadarff(Path(__file__).resolve().parent / "messidor_features.arff")

    # Converting the dataset to a pandas dataframe
    drd_df = pd.DataFrame(drd_dataset[0])

    # Converting non-numeric values to integers datatype
    drd_df = drd_df.apply(pd.to_numeric)

    # Setting up columns
    drd_df.columns = [
        "q",
        "ps",
        "ma05",
        "ma06",
        "ma07",
        "ma08",
        "ma09",
        "ma10",
        "ex05",
        "ex06",
        "ex07",
        "ex08",
        "ex09",
        "ex1a",
        "ex1b",
        "ex1c",
        "dd",
        "dm",
        "amfm",
        "Class",
    ]

    # Making 'Class' as labels data, y
    # Dropping 'Class', creating features data X
    y = drd_df["Class"]
    X = drd_df.drop(["Class"], axis=1)

    return X, y


def validate_aucs(aucs: pd.DataFrame) -> None:
    aucs = aucs.rename(columns=lambda s: s.lower())
    colnames = sorted(aucs.columns.to_list())
    assert colnames == ["auc", "feature"]
    assert aucs.dtypes["auc"] == "float64"
    assert (aucs.dtypes["feature"] == "object") or (aucs.dtypes["feature"] == "string")


def question1() -> None:
    """Calculate and save the k-fold error rates of multiple classifiers on the given (MNIST) data."""

    # Loading the dataset
    X, y = process_mnist_train()
    # Creating index and columns for scores dataframe
    clf_index = ["cnn", "knn1", "knn5", "knn10"]

    # Setting up classifiers with defined parameters
    classifiers = [
        cnn_q1_model(),
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=5),
        KNeighborsClassifier(n_neighbors=10),
    ]

    kfold_scores = calculate_kfold_scores(X, y, clf_index, classifiers, False)
    save_kfold(kfold_scores, question=1)


def question2() -> None:
    """Calculate AUC values for the collected dataset, sorted according to separability."""

    # Getting the features and target data for the dataset
    X, y = process_data()
    auc_values: Dict[str, float] = {}

    # Calculating the ROC AUC score for each feature
    for column in X.columns:
        auc_values[column] = roc_auc_score(y, X[column])

    # Sort the AUC values in terms of separability
    for feature in auc_values:
        if auc_values[feature] < 0.5:
            auc_values[feature] = 1 - auc_values[feature]

    auc_values_list = sorted(auc_values.items(), key=lambda auc: auc[1], reverse=True)

    # Converting AUCs list to dataframe
    auc_df = pd.DataFrame(auc_values_list, columns=["Feature", "AUC"])

    # Verify sorting of AUCs
    validate_aucs(auc_df)

    # Print the top ten AUC values rounded to 3 decimal places, in markdown format
    print(auc_df.head(10).round(3).to_markdown(index=False))

    outfile = Path(__file__).resolve().parent / "aucs.json"
    auc_df.to_json(outfile)
    print(f"AUC values for data successfully saved to {outfile}")


def question3() -> None:
    """Calculate and save the k-fold error rates of multiple classifiers on the collected data."""
    # Getting the features and target data for the dataset
    X, y = process_data()
    # Creating index and columns for scores dataframe
    clf_index = ["ann1", "ann2", "ann3", "ann4", "knn1", "knn5", "knn10"]

    # Setting up classifiers with defined parameters
    classifiers = [
        MLPClassifier(hidden_layer_sizes=5, activation="tanh", solver="lbfgs", random_state=1),
        MLPClassifier(hidden_layer_sizes=40, activation="tanh", solver="adam", max_iter=10000, random_state=1),
        MLPClassifier(hidden_layer_sizes=70, activation="tanh", solver="adam", max_iter=10000, random_state=1),
        MLPClassifier(hidden_layer_sizes=45, activation="tanh", solver="adam", max_iter=10000, random_state=1),
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=5),
        KNeighborsClassifier(n_neighbors=10),
    ]

    kfold_scores = calculate_kfold_scores(X, y, clf_index, classifiers, False)
    save_kfold(kfold_scores, question=3)


def question4() -> None:
    """Calculate and save the k-fold error rates of multiple classifiers on the given (MNIST) data."""

    # Loading the dataset
    X, y = process_mnist_train()

    # Creating index and columns for scores dataframe
    clf_index = ["cnn"]

    # Setting up classifier
    classifiers = [cnn_best_model()]

    kfold_scores = calculate_kfold_scores(X, y, clf_index, classifiers, True)
    save_kfold(kfold_scores, question=4)


if __name__ == "__main__":
    question1()
    question2()
    question3()
    question4()
