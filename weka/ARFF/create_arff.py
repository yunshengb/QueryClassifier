import os

DATA_ROOT = '../../data/'
classes = []

for c in os.listdir(DATA_ROOT):
        if not c.startswith('.'):
            classes.append(c.split(".")[0])

print '@relation _QueryClassifier_data'
print 
print '@attribute queryclass {' + ','.join(classes) + '}'
print '@attribute text string'
print
print '@data'

for c in classes:
    lines = [line.rstrip('\n') for line in open(DATA_ROOT + c + '.txt')]
    for i in lines:
        # print "'" + i.replace("'", "\\'") + "'," + c
        print c + ",'" + i.replace("'", "\\'") + "'"
