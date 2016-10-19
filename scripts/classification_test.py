from __future__ import print_function
import argparse
from sklearn_functions import *
from sklearn.datasets import load_iris


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nf", "--noisy-features", action='store_true', help="Noisy features", required=False)
    parser.add_argument("-nn", "--noise-normal", action='store_true', help="Normal noise", required=False)
    parser.add_argument("-cv", "--cross-validation", type=int, default=5,
                        help="Cross-validation factor", required=False)
    parser.add_argument("-d", "--debug", action='store_true', help="Debugger; run with iris", required=False)
    args = vars(parser.parse_args())
    noise_normal = args['noise_normal']
    noisy_features = args['noisy_features']
    args_cv = args['cross_validation']

    # Set seed
    np.random.seed(0)

    """ Classification """
    if args['debug']:
        # Add data
        dataset = load_iris()  # classification example
        X_clas, y_clas = dataset.data, dataset.target

        # Add noise -- to add: add argparse for degree and rand_state
        if noise_normal:
            X_clas = add_noise_normal(X_clas, degree=1)
        if noisy_features:
            X_clas = add_noisy_features(X_clas, degree=200, rand_state=1)

        # Binarize
        X_clas_bin, y_clas_bin = X_clas[y_clas != 2], y_clas[y_clas != 2]  # y = 0 or 1 only

        """ Generate classifiers w/o cross-validation """
        print("Classifiers:")
        clas_models = train_without_cv_all(X_clas_bin, y_clas_bin)

        """ Generate classifiers w/ cross-validation and ROC """
        classifiers = classification_models()
        for k, model in classifiers.items():
            plot_ROC(model, X_clas_bin, y_clas_bin, cv_fold=args_cv, png_filename=str(k))


        # TODO ROC for multiclass classifier and ROC for regression models

        """ One vs Rest Classifier """
        # # Binarize the output
        # y_clas = label_binarize(y_clas, classes=[0, 1, 2])
        # n_classes = y_clas.shape[1]
        # n_samples, n_features = X_clas.shape
        # random_state = np.random.RandomState(0)
        # classifier = OneVsRestClassifier(svm.SVC(kernel='linear',
        #                                          probability=True,
        #                                          random_state=random_state))
        # # split and score
        # X_train, X_test, y_train, y_test = train_test_split(X_clas, y_clas, test_size=.5)
        # y_score = classifier.fit(X_train, y_train).decision_function(X_test)
