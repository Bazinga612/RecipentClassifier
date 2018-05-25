from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score
# naive bayes model
def train_naive_bayes(x_train_seq,train_y,x_test_seq,test_y):
    gnb = GaussianNB()
    gnb.fit(x_train_seq, train_y)
    print(gnb)
    # make predictions
    predicted = gnb.predict(x_test_seq)
    print(accuracy_score(test_y, predicted))
    scores = cross_val_score(gnb, x_train_seq, train_y, cv=5)
    print("Cross validation accuracy is:", scores.mean()*100)
    print("Maximum accuracy for Gaussian Naive Bayes Model is:", accuracy_score(test_y, predicted)*100)

    mnb = MultinomialNB()
    x = 1.0
    max_alpha = 0
    max_accuracy = 0
    for values in range(5):
        MultinomialNB(alpha=x, class_prior=None, fit_prior=True)
        mnb.fit(x_train_seq, train_y)
        predicted1 = mnb.predict(x_test_seq)

        if accuracy_score(test_y, predicted1) > max_alpha:
            max_alpha = x
            max_accuracy = accuracy_score(test_y, predicted1)
        x = x + 1.0

    print("Maximum accuarcy for Multinomial Naive Bayes Model with maximum alpha", max_alpha, "is:", max_accuracy*100)
    plt.figure()

    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        gnb, x_train_seq, train_y, cv=None, n_jobs=1)
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
