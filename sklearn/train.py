import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
import cPickle

NEWLINE = '\n'

GENERAL = 'general'
CALENDAR = 'calendar'
IMM = "imm"

SOURCES = [
    ('../data/calendar/calendar_questions.txt',         CALENDAR),
    ('../data/general/questions_80.txt',                GENERAL),
    ('../data/imm/imm_questions.txt',                IMM)
]

SKIP_FILES = {'cmds'}

def build_data_frame(path, classification):
    lines = [line.rstrip('\n') for line in open(path)]
    rows = []
    index = []
    for text in lines:
        rows.append({'text': text, 'class': classification})
        index.append(text)

    data_frame = DataFrame(rows, index=index)
    return data_frame


data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))




data = data.reindex(numpy.random.permutation(data.index))

pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
    ('classifier',         MultinomialNB())
])

k_fold = KFold(n=len(data), n_folds=3)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values.astype(str)

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values.astype(str)

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

#     confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label=CALENDAR)
    scores.append(score)

print('Total documents classified:', len(data))
print('Score:', sum(scores)/len(scores))
# print('Confusion matrix:')
# print(confusion)
 
# save the classifier
with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(pipeline, fid)