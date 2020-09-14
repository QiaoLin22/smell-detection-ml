res = {}
def __init__(self, db):
    self.db = db
def compute(db):
    from scipy.io import arff

    import pandas as pd

    data = arff.loadarff(db)

    df = pd.DataFrame(data[0])
    X = df.iloc[:, :-1].values
    Y_data = df.iloc[:, -1].values
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.preprocessing import LabelEncoder
    lb_make = LabelEncoder()
    y = lb_make.fit_transform(Y_data)
    # y = MultiLabelBinarizer().fit_transform(Y_data)
    from sklearn.tree import DecisionTreeClassifier
    tree_clf = DecisionTreeClassifier()
    # Feature Engineering

    # 1.Imputation
    from sklearn.impute import SimpleImputer

    X_copy = df.iloc[:, :-1].copy()

    imputer = SimpleImputer(strategy="median")

    imputer.fit(X_copy)

    new_X = imputer.transform(X_copy)

    new_X_df = pd.DataFrame(new_X, columns=X_copy.columns, index=X_copy.index)
    # Scaling Standarization
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(new_X)
    new_X = scaler.transform(new_X)

    # Feature selection, removing features with low variance¶
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    new_X = sel.fit_transform(new_X)
    # Training
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.15, random_state=42)
    from sklearn.model_selection import cross_val_score

    # grid search DT
    from sklearn.model_selection import GridSearchCV

    import numpy as np

    depths = np.arange(1, 11)

    num_leafs = [1, 5, 10, 20, 50]

    param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': depths, 'min_samples_leaf': num_leafs}

    new_tree_clf = DecisionTreeClassifier()

    grid_search_acc = GridSearchCV(new_tree_clf, param_grid, cv=10, scoring="accuracy", return_train_score=True)
    grid_search_acc.fit(X_train, y_train)
    grid_search_f1 = GridSearchCV(new_tree_clf, param_grid, cv=10, scoring="f1_micro", return_train_score=True)
    grid_search_f1.fit(X_train, y_train)
    # evaluation
    from sklearn.metrics import accuracy_score

    best_model_acc = grid_search_acc.best_estimator_
    best_model_f1 = grid_search_f1.best_estimator_
    tree_clf.fit(X_train, y_train)

    # Train and Test accuracy
    dt_train_acc = accuracy_score(y_train, best_model_acc.predict(X_train))
    dt_test_acc = accuracy_score(y_test, best_model_acc.predict(X_test))
    dt_train_f1 = accuracy_score(y_train, best_model_f1.predict(X_train))
    dt_test_f1 = accuracy_score(y_test, best_model_f1.predict(X_test))

    res = {}
    res[db+"_decision_tree_train_acc"] = dt_train_acc
    res[db+"_decision_tree_test_acc"] = dt_test_acc
    res[db+"_decision_tree_train_f1"] = dt_train_f1
    res[db+"_decision_tree_test_f1"] = dt_test_f1
    # Grid Search RF
    from sklearn.ensemble import RandomForestClassifier

    rf_param_grid = {
        'n_estimators': [20, 40, 60, 80, 100],
        'max_depth': [2, 5, 7],
        'min_samples_leaf': [1, 2, 4]}
    rf = RandomForestClassifier()
    rf_grid_search_acc = GridSearchCV(rf, rf_param_grid, cv=3, scoring="accuracy", return_train_score=True)
    rf_grid_search_acc.fit(X_train, y_train)
    rf_grid_search_f1 = GridSearchCV(rf, rf_param_grid, cv=3, scoring="f1_micro", return_train_score=True)
    rf_grid_search_f1.fit(X_train, y_train)
    # Evaluation
    rf_best_model_acc = rf_grid_search_acc.best_estimator_
    rf_best_model_f1 = rf_grid_search_f1.best_estimator_
    rf_train_acc = accuracy_score(y_train, rf_best_model_acc.predict(X_train))
    rf_test_acc = accuracy_score(y_test, rf_best_model_acc.predict(X_test))
    rf_train_f1 = accuracy_score(y_train, rf_best_model_f1.predict(X_train))
    rf_test_f1 = accuracy_score(y_test, rf_best_model_f1.predict(X_test))
    res[db+"_random_forest_train_acc"] = rf_train_acc
    res[db+"_random_forest_test_acc"] = rf_test_acc
    res[db+"_random_forest_train_f1"] = rf_train_f1
    res[db+"_random_forest_test_f1"] = rf_test_f1
    # CV Grid Search
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb_params = {}
    nb_grid_search_acc = GridSearchCV(nb, nb_params, cv=3, scoring="accuracy", return_train_score=True)
    nb_grid_search_f1 = GridSearchCV(nb, nb_params, cv=3, scoring="f1_micro", return_train_score=True)
    nb_grid_search_acc.fit(X_train, y_train)
    nb_grid_search_f1.fit(X_train, y_train)
    nb_best_model_acc = nb_grid_search_acc.best_estimator_
    nb_best_model_f1 = nb_grid_search_f1.best_estimator_
    nb_train_acc = accuracy_score(y_train, nb_best_model_acc.predict(X_train))
    nb_test_acc = accuracy_score(y_test, nb_best_model_acc.predict(X_test))
    nb_train_f1 = accuracy_score(y_train, nb_best_model_f1.predict(X_train))
    nb_test_f1 = accuracy_score(y_test, nb_best_model_f1.predict(X_test))
    res[db+"_naive_bayes_train_acc"] = nb_train_acc
    res[db+"_naive_bayes_test_acc"] = nb_test_acc
    res[db+"_naive_bayes_train_f1"] = nb_train_f1
    res[db+"_naive_bayes_test_f1"] = nb_test_f1
    # SVC Grid Search
    # linear
    from sklearn import svm
    linear = svm.SVC(kernel='linear')
    parameters = {'C': [1, 10]}
    linear_grid_search_acc = GridSearchCV(linear, parameters, cv=3, scoring="accuracy", return_train_score=True)
    linear_grid_search_f1 = GridSearchCV(linear, parameters, cv=3, scoring="f1_micro", return_train_score=True)
    linear_grid_search_acc.fit(X_train, y_train)
    linear_grid_search_f1.fit(X_train, y_train)
    linear_best_model_acc = linear_grid_search_acc.best_estimator_
    linear_best_model_f1 = linear_grid_search_f1.best_estimator_
    linear_train_acc = accuracy_score(y_train, linear_best_model_acc.predict(X_train))
    linear_test_acc = accuracy_score(y_test, linear_best_model_acc.predict(X_test))
    linear_train_f1 = accuracy_score(y_train, linear_best_model_f1.predict(X_train))
    linear_test_f1 = accuracy_score(y_test, linear_best_model_f1.predict(X_test))
    res[db+"_svc_linear_train_acc"] = linear_train_acc
    res[db+"_svc_linear_test_acc"] = linear_test_acc
    res[db+"_svc_linear_train_f1"] = linear_train_f1
    res[db+"_svc_linear_test_f1"] = linear_test_f1
    # polynomial
    poly = svm.SVC(kernel='poly')
    parameters = {'C': [1, 10]}
    poly_grid_search_acc = GridSearchCV(poly, parameters, cv=3, scoring="accuracy", return_train_score=True)
    poly_grid_search_f1 = GridSearchCV(poly, parameters, cv=3, scoring="f1_micro", return_train_score=True)
    poly_grid_search_acc.fit(X_train, y_train)
    poly_grid_search_f1.fit(X_train, y_train)
    poly_best_model_acc = poly_grid_search_acc.best_estimator_
    poly_best_model_f1 = poly_grid_search_f1.best_estimator_
    poly_train_acc = accuracy_score(y_train, poly_best_model_acc.predict(X_train))
    poly_test_acc = accuracy_score(y_test, poly_best_model_acc.predict(X_test))
    poly_train_f1 = accuracy_score(y_train, poly_best_model_f1.predict(X_train))
    poly_test_f1 = accuracy_score(y_test, poly_best_model_f1.predict(X_test))
    res[db+"_svc_poly_train_acc"] = poly_train_acc
    res[db+"_svc_poly_test_acc"] = poly_test_acc
    res[db+"_svc_polytrain_f1"] = poly_train_f1
    res[db+"_svc_poly_test_f1"] = poly_test_f1
    # rbf
    rbf = svm.SVC(kernel='rbf')
    parameters = {'C': [1, 10]}
    rbf_grid_search_acc = GridSearchCV(rbf, parameters, cv=3, scoring="accuracy", return_train_score=True)
    rbf_grid_search_f1 = GridSearchCV(rbf, parameters, cv=3, scoring="f1_micro", return_train_score=True)
    rbf_grid_search_acc.fit(X_train, y_train)
    rbf_grid_search_f1.fit(X_train, y_train)
    rbf_best_model_acc = rbf_grid_search_acc.best_estimator_
    rbf_best_model_f1 = rbf_grid_search_f1.best_estimator_
    rbf_train_acc = accuracy_score(y_train, rbf_best_model_acc.predict(X_train))
    rbf_test_acc = accuracy_score(y_test, rbf_best_model_acc.predict(X_test))
    rbf_train_f1 = accuracy_score(y_train, rbf_best_model_f1.predict(X_train))
    rbf_test_f1 = accuracy_score(y_test, rbf_best_model_f1.predict(X_test))
    res[db+"_svc_rbf_train_acc"] = rbf_train_acc
    res[db+"_svc_rbf_test_acc"] = rbf_test_acc
    res[db+"_svc_rbf_train_f1"] = rbf_train_f1
    res[db+"_svc_rbf_test_f1"] = rbf_test_f1
    # sigmoid
    sigmoid = svm.SVC(kernel='sigmoid')
    parameters = {'C': [1, 10]}
    sigmoid_grid_search_acc = GridSearchCV(sigmoid, parameters, cv=3, scoring="accuracy", return_train_score=True)
    sigmoid_grid_search_f1 = GridSearchCV(sigmoid, parameters, cv=3, scoring="f1_micro", return_train_score=True)
    sigmoid_grid_search_acc.fit(X_train, y_train)
    sigmoid_grid_search_f1.fit(X_train, y_train)
    sigmoid_best_model_acc = sigmoid_grid_search_acc.best_estimator_
    sigmoid_best_model_f1 = sigmoid_grid_search_f1.best_estimator_
    sigmoid_train_acc = accuracy_score(y_train, sigmoid_best_model_acc.predict(X_train))
    sigmoid_test_acc = accuracy_score(y_test, sigmoid_best_model_acc.predict(X_test))
    sigmoid_train_f1 = accuracy_score(y_train, sigmoid_best_model_f1.predict(X_train))
    sigmoid_test_f1 = accuracy_score(y_test, sigmoid_best_model_f1.predict(X_test))
    res[db+"_svc_sigmoid_train_acc"] = sigmoid_train_acc
    res[db+"_svc_sigmoid_test_acc"] = sigmoid_test_acc
    res[db+"_svc_sigmoidtrain_f1"] = sigmoid_train_f1
    res[db+"_svc_sigmoid_test_f1"] = sigmoid_test_f1

    # result
    return res


