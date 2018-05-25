from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def train_decision_tree(x_train_seq,x_test_seq,train_y,test_y):

    dtree_model = DecisionTreeClassifier(max_depth=1).fit(x_train_seq, train_y)
    dtree_predictions = dtree_model.predict(x_test_seq)


    # summarize the fit of the model

    print("Maximum accuarcy for decision tree is:",(accuracy_score(test_y, dtree_predictions)) * 100)
    train_sizes, train_scores, test_scores = learning_curve(
        dtree_model, x_train_seq, train_y, cv=None, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()