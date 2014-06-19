
def run_log_model(xi,y,add_intercept=True):
    import sklearn.linear_model as lm
    import numpy as np
    import statsmodels.api as sm
    from sklearn.cross_validation import StratifiedShuffleSplit

    a = lm.LogisticRegression()
    k = len(xi)
    X = xi[0]

    for i in range(1,k):
        X = np.column_stack((X,xi[i]))
    if add_intercept:
        X = sm.add_constant(X)

    a.fit(X,y)
    
    #ROC
    sss = StratifiedShuffleSplit(y, 100, test_size=0.5, random_state=0)
    train_indices, test_indices = [], []
    for tr,te in sss:
        train_indices.append(tr)
        test_indices.append(te)

    roc_info = map(get_ROC,[X]*100,[y]*100,train_indices,test_indices)
    roc_info = np.asarray(roc_info)
    fpr = roc_info[:,0]
    tpr = roc_info[:,1]
    auc = roc_info[:,2]

    return np.mean(auc)

def get_ROC(X,y,train_index,test_index):
    from sklearn.metrics import roc_curve, auc
    from sklearn.utils import shuffle
    from sklearn.cross_validation import StratifiedShuffleSplit
    import sklearn.linear_model as lm
    import numpy as np
    #X, y = shuffle(X, y)
    #sss = StratifiedShuffleSplit(y, n, test_size=0.5, random_state=0)
    #half = int(len(y) / 2)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Run classifier
    classifier = lm.LogisticRegression()
    classifier.fit(X_train, y_train)
    probas_ = classifier.predict_proba(X_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    if np.isfinite(fpr).all() and np.isfinite(tpr).all():
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = 0
        print "nans"
    return fpr,tpr,roc_auc

#
