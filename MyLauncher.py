##
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import pearsonr
from MyModel import STSModels
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor

from nltk import download
from nltk.corpus import treebank
from nltk.tag.perceptron import PerceptronTagger
#download('treebank')

def ptTrain():
    train_data = treebank.tagged_sents()
    #Perceptron tagger
    per = PerceptronTagger(load='false')
    per.train(train_data)
    return per

filenames_train = ['MSRpar', 'MSRvid', 'SMTeuroparl']
filenames_test = ['MSRpar', 'MSRvid', 'SMTeuroparl',
				  'surprise.OnWN', 'surprise.SMTnews']
##
#print('Train PosTagger')
per = ''#ptTrain()

for test_file in filenames_test:
	print(test_file)
	filename_test = [test_file]
	## 
	#############################################################
	print()
	print('#####################################')
	print()
	# Lemmas

	print('Train of Lemmas Model')
	trn = STSModels()
	trn.fit(filenames_train, ['lemmas'], 'train', per)
	##
	X_lem = trn.X
	regr = AdaBoostRegressor().fit(trn.X, trn.y)
	#regr = LogisticRegression(solver='lbfgs',
	#                          multi_class='auto').fit(trn.X, trn.y)
	##
	print('Test of Lemmas Model')
	tst = STSModels()
	tst.fit(filename_test, ['lemmas'], 'test', per)
	Xtst_lemma = tst.X
	##
	print('Prediction')
	pred = regr.predict(tst.X)

	result = pearsonr(pred, tst.y)[0] # taking just the Pearson coeficient
	print(result)

	## 
	#############################################################
	print()
	print('#####################################')
	print()
	# Lesk
	print('Train of Lesk Model')
	trn = STSModels()
	trn.fit(filenames_train, ['lesk'], 'train', per)
	X_lesk = trn.X
	##
	regr = AdaBoostRegressor().fit(trn.X, trn.y)
	##
	print('Test of Lesk Model')
	tst = STSModels()
	tst.fit(filename_test, ['lesk'], 'test', per)
	Xtst_lesk = tst.X
	##
	print('Prediction of Lesk Model')
	pred = regr.predict(tst.X)

	result = pearsonr(pred, tst.y)[0] # taking just the Pearson coeficient
	print(result)
	##

	##
	#############################################################
	print()
	print('#####################################')
	print()
	# Synsets
	print('Train Synsets')
	trn = STSModels()
	trn.fit(filenames_train, ['synsets'], 'train', per)
	##
	X_syn = trn.X
	regr = AdaBoostRegressor().fit(trn.X, trn.y)
	##
	print('Test of Synsets Model')
	tst = STSModels()
	tst.fit(filename_test, ['synsets'], 'test', per)
	Xtst_syn = tst.X
	##
	print('Prediction of Synsets')
	pred = regr.predict(tst.X)

	result = pearsonr(pred, tst.y)[0] # taking just the Pearson coeficient
	print(result)

	#############################################################
	print()
	print('#####################################')
	print()
	## Lemmas + Lesk
	print('Lemmas + Lesk + Synsets')
	X = np.concatenate([X_lem, X_lesk, X_syn], axis=1)
	regr = AdaBoostRegressor().fit(X, trn.y)
	print('Prediction of Lemmas + Lesk + Synsets')
	X_tst = np.concatenate([Xtst_lesk, Xtst_lemma, Xtst_syn], axis=1)
	pred = regr.predict(X_tst)

	result = pearsonr(pred, tst.y)[0] # taking just the Pearson coeficient
	print(result)
	#print(tst.y - pred)
	print()
	print(X.shape, len(trn.y))
	print("-------------------------------------------")
	print()
