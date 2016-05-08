cd sklearn
/Users/yba/anaconda/bin/python2.7 train.py
cd ..
cd weka/ARFF
python create_arff.py > ../data.arff
cd ..
ant > /dev/null
java -cp :weka-stable-3.6.13.jar MyFilteredLearner data.arff myClassifier.dat
