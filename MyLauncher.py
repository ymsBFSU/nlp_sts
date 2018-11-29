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
from sklearn.pipeline import Pipeline
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

#print('Train PosTagger')
#per = ptTrain()
regressor = LinearRegression()
'''
model = Pipeline(steps=[
        ('lem', STSModels(['lemmas'])),
        ('lesk', STSModels(['lesk']))
    ])
'''
model_selection = ['bow']#['lemmas', 'lesk', 'synsets', 'ngrams','bow']
##
for test_file in filenames_test:
    print(test_file)
    filename_test = [test_file]

    #############################################################
    if 'bow' in model_selection:
        print()
        print('#####################################')
        print()
        # Bow
        print('Train of Bow Model')
        trn = STSModels(['bow'])
        trn.fit(filenames_train, 'train')
        X_gram = trn.X
    #############################################################
    if 'ngrams' in model_selection:
        print()
        print('#####################################')
        print()
        # 2-grams
        print('Train of Grams Model')
        trn = STSModels(['ngrams'])
        trn.fit(filenames_train, 'train')
        X_gram = trn.X

        ##
        print('Test of Grams Model')
        tst = STSModels(['ngrams'])
        tst.transform(filename_test, 'test')
        Xtst_gram = tst.X
        ##
        print('Prediction of Grams')
        regr_gram = regressor.fit(trn.X, trn.y)
        # regr = LogisticRegression(solver='lbfgs',
        #                          multi_class='auto').fit(trn.X, trn.y)
        pred_gram = regr_gram.predict(tst.X)

        result = pearsonr(pred_gram, tst.y)[0] # taking just the Pearson coeficient
        print(result)
    #############################################################
    if 'lemmas' in model_selection:
        print()
        print('#####################################')
        print()
        # Lemmas

        print('Train of Lemmas Model')
        trn = STSModels(['lemmas'])
        trn.fit(filenames_train, 'train')
        X_lem = trn.X

        ##
        print('Test of Lemmas Model')
        tst = STSModels(['lemmas'])
        tst.transform(filename_test, 'test')
        Xtst_lem = tst.X
        ##
        print('Prediction of Lemmas')
        regr_lem = regressor.fit(trn.X, trn.y)
        # regr = LogisticRegression(solver='lbfgs',
        #                          multi_class='auto').fit(trn.X, trn.y)
        pred_lem = regr_lem.predict(tst.X)

        result = pearsonr(pred_lem, tst.y)[0] # taking just the Pearson coeficient
        print(result)

    ## 
    #############################################################
    if 'lesk' in model_selection:
        print()
        print('#####################################')
        print()
        # Lesk
        print('Train of Lesk Model')
        trn = STSModels(['lesk'])
        trn.fit(filenames_train, 'train')
        X_lesk = trn.X

        ##
        print('Test of Lesk Model')
        tst = STSModels(['lesk'])
        tst.transform(filename_test, 'test')
        Xtst_lesk = tst.X

        ##
        print('Prediction of Lesk Model')
        regr_lesk = regressor.fit(trn.X, trn.y)
        pred_lesk = regr_lesk.predict(tst.X)

        result = pearsonr(pred_lesk, tst.y)[0] # taking just the Pearson coeficient
        print(result)

    ##
    #############################################################
    if 'synsets' in model_selection:
        print()
        print('#####################################')
        print()
        # Synsets
        print('Train Synsets')
        trn = STSModels(['synsets'])
        trn.fit(filenames_train, 'train')
        X_syn = trn.X

        ##
        print('Test of Synsets Model')
        tst = STSModels(['synsets'])
        tst.transform(filename_test, 'test')
        Xtst_syn = tst.X

        ##
        print('Prediction of Synsets')
        regr = regressor.fit(trn.X, trn.y)
        pred = regr.predict(tst.X)
        result = pearsonr(pred, tst.y)[0] # taking just the Pearson coeficient
        print(result)

    #############################################################
    if 'all' in model_selection:
        print()
        print('#####################################')
        print()
        ## Lemmas + Lesk
        print('Lemmas + Lesk + Synsets')
        X = np.concatenate([X_lem, X_lesk, X_syn], axis=1)
        regr = regressor.fit(X, trn.y)

        print('Prediction of Lemmas + Lesk + Synsets')
        X_tst = np.concatenate([Xtst_lesk, Xtst_lem, Xtst_syn], axis=1)
        pred = regr.predict(X_tst)

        result = pearsonr(pred, tst.y)[0] # taking just the Pearson coeficient
        print(result)
        #print(tst.y - pred)
        print()
        print(X.shape, len(trn.y))
        print("-------------------------------------------")
        print()
