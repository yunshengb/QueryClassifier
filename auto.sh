cd sklearn
~/anaconda/bin/python2.7 train.py
cd ..
cd weka/ARFF
python create_arff.py > ../data.arff
cd ..
ant > /dev/null
java -classpath :lib/snowball-20051019.jar:lib/weka.jar MyFilteredLearner data.arff myClassifier.dat
