import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
import cPickle

ASR_QA = 'ASR_QA'
ASR_IMM = 'ASR_IMM'
ASR_IMM_QA = 'ASR_IMM_QA'
ASR_CA = 'ASR_CA'

SOURCES = [
    ('../data/ASR_QA.txt',                 ASR_QA),
    ('../data/ASR_IMM.txt',                ASR_IMM),
    ('../data/ASR_IMM_QA.txt',             ASR_IMM_QA),
    ('../data/ASR_CA.txt',                 ASR_CA)
]

def build_data_frame(path, classification):
    lines = [line.rstrip('\n') for line in open(path)]
    rows = []
    index = []
    for text in lines:
        if text in index:
            print 'duplicate in ' + path + ": " + text
            exit(1)
        rows.append({'text': text, 'class': classification})
        index.append(text)

    data_frame = DataFrame(rows, index=index)
    return data_frame

data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))

data = data.reindex(numpy.random.permutation(data.index))

pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range = (1, 2))),
#     ('classifier',         PassiveAggressiveClassifier())
    ('classifier',         LinearSVC()) 
])

k_fold = KFold(n=len(data), n_folds=6)
scores = []
confusion = numpy.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
for train_indices, test_indices in k_fold:
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values.astype(str)

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values.astype(str)

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, average='weighted')
    scores.append(score)

print('************************* Scikit Learn *************************')
print('Total documents classified:', len(data))
print('Score:', sum(scores) / len(scores))
print('Confusion matrix:')
print(confusion)
 
# save the classifier
with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(pipeline, fid)
