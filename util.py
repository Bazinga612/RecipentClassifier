import json
import random
import nltk
from nltk import FreqDist
from numpy import array
from keras.utils import to_categorical
import numpy as np
import naive_bayes_model
import sophisticated_model_nn
import decision_tree_model

names=['Tom','Richard','Alicia','Will','Tracie','George','Lisa','Lyn','Tim','Mike','Cody','Emma','Austin','Sidney','Shane','Korey',
           'Alexander','Kirby','Michelle','Nicolle','Ramon','Connor','Johnathon','Timothy','Christian']

initial=['Can you tell x to','Can you inform x to','Message x to','Can you call and tell x to','Ping x that','Can you send x a message that']
end=['can you tell that to x .','inform x .','message x .']

test_names=['Beth','Holly','Kirk','John','Serena','Leon','Jason','Jerry']
test_initial=['Can you ring x up and tell that','Remind x to','Will you tell x to','Will you message x to']
test_end=['send email to x .']


def create_dataset():
    i=0
    template=[]
    with open('mappingv3.json') as f:
        data = json.load(f)
        for i in range(len(data)):
            template.append(data[i]["nl_command_statment"])
            i=i+1

    formed_template = form_template(template)
    train = []
    test = []

    train_template = formed_template[:40]

    template2 = list(set(train_template).symmetric_difference(formed_template))
    test_template = template2[:10]

    for values_initial in initial:

        for values_template in train_template:
            value_names = random.sample(names, 1)[0]
            names_template = values_initial.replace("x", value_names)
            train.append(names_template + " " + values_template)

    for value_end in end:

        for value_template in train_template:
            value_names = random.sample(names, 1)[0]
            names_template = value_end.replace("x", value_names)
            value_template = value_template.replace(".", "")
            train.append(value_template + ", " + names_template)


    for values_initial in test_initial:
        value_names = random.sample(test_names, 1)[0]
        names_template = values_initial.replace("x", value_names)
        for values_template in test_template:
            test.append(names_template + " " + values_template)

    for value_end in test_end:
        value_names = random.sample(test_names, 1)[0]
        names_template = value_end.replace("x", value_names)
        for value_template in test_template:
            test.append(value_template + " " + names_template)

    return train,test


def form_template(template):
    template1 = []
    for value in template:
        sent = nltk.sent_tokenize(value)[0].lower()
        sent_tok = nltk.word_tokenize(sent)

        if nltk.pos_tag(sent_tok)[0][1] == 'VB':
            template1.append(sent)
    return template1


def generate_label_index(data, names):
    final_list = []
    for values in data:
        list1 = values.split()
        m = list1.index(list(set(names).intersection(list1))[0])
        final_list.append(m)
    return final_list

train,test=create_dataset()


train_y = generate_label_index(train, names)
test_y = generate_label_index(test, test_names)

def train_embedddings():
    X_train = [x.split()[:-1] for x in train]
    X_test = [x.split()[:-1] for x in test]

    x_distr = FreqDist(np.concatenate(X_train + X_test))
    x_vocab = x_distr.most_common(min(len(x_distr), 10000))

    x_idx2word = [word[0] for word in x_vocab]

    x_idx2word.insert(0, '<PADDING>')
    x_idx2word.append('<NA>')

    x_word2idx = {word: idx for idx, word in enumerate(x_idx2word)}

    x_train_seq = np.zeros((len(X_train), 45), dtype=np.int32)
    for i, da in enumerate(X_train):
        for j, token in enumerate(da):
            # truncate long Titles
            if j >= 45:
                break

            # represent each token with the corresponding index
            if token in x_word2idx:
                x_train_seq[i][j] = x_word2idx[token]
            else:
                x_train_seq[i][j] = x_word2idx['<NA>']

    x_test_seq = np.zeros((len(X_test), 45),dtype=np.int32)  # padding implicitly present, as the index of the padding token is 0

    # form embeddings for samples testing data
    for i, da in enumerate(X_test):
        for j, token in enumerate(da):
            # truncate long Titles
            if j >= 45:
                break

            # represent each token with the corresponding index
            if token in x_word2idx:
                x_test_seq[i][j] = x_word2idx[token]
            else:
                x_test_seq[i][j] = x_word2idx['<NA>']

    return x_train_seq,x_test_seq,x_vocab

x_train_seq,x_test_seq,x_vocab=train_embedddings()

def one_hot_encoding():
    y_train_seq = np.zeros((len(train_y), 45), dtype=np.int32)
    y_test_seq = [[]] * len(test_y)
    for counter, values in enumerate(train_y):
        y_train_seq[counter] = [0] * 45
        index = train_y[counter]
        y_train_seq[counter][index] = 1

    for counter, values in enumerate(test_y):
        y_test_seq[counter] = [0] * 45
        index = test_y[counter]
        y_test_seq[counter][index] = 1

    x_train1_seq = []
    for counter, values in enumerate(x_train_seq):
        x_train1_seq.append(to_categorical(x_train_seq[counter], num_classes=(len(x_vocab) + 1)))
    x_test1_seq = []
    for counter, values in enumerate(x_test_seq):
        x_test1_seq.append(to_categorical(values, num_classes=(len(x_vocab) + 1)))
    y_train1_seq = []
    for counter, values in enumerate(y_train_seq):
        y_train1_seq.append(to_categorical(values, num_classes=(len(x_vocab) + 1)))

    x_test1_seq = array(x_test1_seq)
    x_train1_seq = array(x_train1_seq)
    y_train1_seq = array(y_train1_seq)
    return y_train_seq,y_test_seq

y_train_seq,y_test_seq=one_hot_encoding()



def invoke_classifier(input):
    if input == 1:
        naive_bayes_model.train_naive_bayes(x_train_seq,train_y,x_test_seq,test_y)
    elif input == 2:
        decision_tree_model.train_decision_tree(x_train_seq,x_test_seq,train_y,test_y)
    else:
        model=sophisticated_model_nn.train_neural_model(x_vocab,x_train_seq,y_train_seq)
        sophisticated_model_nn.neural_model_eval(model,x_test_seq,y_test_seq)










