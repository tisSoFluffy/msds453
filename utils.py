import re
import string
from os import listdir
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from numpy import array



def load_doc(filename):
    '''Load the file and return the text of the given a filename'''
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def clean_doc(doc):
    '''Remove non-alpha chars, punctuation, and stopwords'''
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def doc_to_line(filename, vocab):
    '''load doc, clean and return line of tokens'''
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)


def process_docs(directory, vocab, is_train):
    '''load all docs in a directory'''
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
    # skip any reviews in the test set
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines

def load_clean_dataset(vocab, is_train):
    '''Load and clean a dataset'''
    neg = process_docs('txt_sentoken/neg', vocab, is_train)
    pos = process_docs('txt_sentoken/pos', vocab, is_train)
    docs = neg + pos
    # prepare labels
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
    return docs, labels

def define_model(n_words):
    '''Define Network'''
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def evaluate_model(Xtrain, ytrain, Xtest, ytest):
    '''Evaluate neural network model '''
    scores = list()
    n_repeats = 10
    n_words = Xtest.shape[1]
    for i in range(n_repeats):
        # define network
        model = define_model(n_words)
        # fit network
        model.fit(Xtrain, ytrain, epochs=10, verbose=0)
        # evaluate
        _, acc = model.evaluate(Xtest, ytest, verbose=0)
        scores.append(acc)
        print('%d accuracy: %s' % ((i+1), acc))
    return scores

def prepare_data(train_docs, test_docs, mode):
    '''Prepare bag of words encoding of docs'''
    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs)
    # encode training data set
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
    # encode training data set
    Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
    return Xtrain, Xtest

def predict_sentiment(review, vocab, tokenizer, model):
    #clean
    tokens = clean_doc(review)
    #filter by vocab
    tokens = [w for w in tokens if w in vocab]
    #convert to a line
    line = ' '.join(tokens)
    #encode
    encoded = tokenizer.texts_to_matrix([line], mode='binary')

    yhat = model.predict(encoded, verbose=0)

    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), 'NEGATIVE'
    return (percent_pos), 'POSITIVE'

def create_tokenizer(lines):
    '''fit a tokenizer'''
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer    

def add_doc_to_vocab(filename, vocab):
    #load doc
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)

def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# load all docs in a directory
def process_docs_to_vocab(directory, vocab):
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
    # add doc to vocab
        add_doc_to_vocab(path, vocab)