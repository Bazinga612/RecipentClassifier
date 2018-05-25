import util

print('There are 3 different classifiers for this task: naive bayes, decision tree or a sophisticated model. Learning curve would appear for decision tree and naive bayes.Close the polts to view the accuracies.')

print('Please enter 1 for naive bayes or 2 for decision tree or 3 for neural network')
user_entered_input = int(input())
util.invoke_classifier(user_entered_input)








