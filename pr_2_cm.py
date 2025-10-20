import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

def func(A, B, n1, n2):
    #dataset
    iris = datasets.load_iris()
    X, y = iris.data[:100, [A, B]], iris.target[:100]
    le = LabelEncoder()
    
    #noise for benchamrking
    X = X + np.random.normal(n1, n2, X.shape)
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.47, random_state=5, stratify=y)

    models = {
        'PPN': Pipeline([('scaler', StandardScaler()), ('clf', Perceptron())]),
        'SVC': Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True))]),
        'Random Forest': Pipeline([('clf', RandomForestClassifier())]),
        'Bagging': Pipeline([('clf', BaggingClassifier())]),
        'AdaBoost': Pipeline([('clf', AdaBoostClassifier())])
    }

    colors = ['red', 'blue', 'yellow', 'black', 'orange']

    #roc curve
    roc_auc = []
    plt.figure(figsize=(8,6))
    for (name, model), c in zip(models.items(), colors):
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            y_preds = model.predict_proba(X_test)[:, 1]
        else:
            y_preds = model.decision_function(X_test)

        # Compute ROC
        fpr, tpr, threshold = roc_curve(y_test, y_preds)
        auc_score = auc(fpr, tpr)
        roc_auc.append([name, auc_score])

        # Plot
        plt.plot(fpr, tpr, label=f'{name}', color=c)

    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.show()

    print("ROC AUC values:", roc_auc)
    return roc_auc
