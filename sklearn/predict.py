import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
import cPickle

# load it again
with open('my_dumped_classifier.pkl', 'rb') as fid:
    pipeline = cPickle.load(fid)

examples = ['What are my future events?', "What is on Johann's calendar?", "Who is the president of China?"]
predictions = pipeline.predict(examples)
print(predictions)