import sys

if len(sys.argv) == 1:
	filename = '/Users/yba/Documents/U/Sirius/SVM/calendar_ham/calendar/calendar_questions.txt'
	# filename = '/Users/yba/Documents/U/Sirius/SVM/calendar_ham/ham/questions_80.txt'
else:
	filename = sys.argv[1]

lines = [line.rstrip('\n') for line in open(filename)]

for i in lines:
	print "calendar,'" + i.replace("'", "\\'") + "'"

