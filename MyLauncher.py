##
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from MyModel import STSModels

from nltk import download
from nltk.corpus import treebank
from nltk.tag.perceptron import PerceptronTagger
download('treebank')

def ptTrain():
    train_data = treebank.tagged_sents()
    #Perceptron tagger
    per = PerceptronTagger(load='false')
    per.train(train_data)
    return per

filename = 'SMTeuroparl'
##
print('Train PosTagger')
per = ptTrain()
##
print('Train of Lemmas Model')
trn = STSModels()
trn.fit_lemmas(filename, 'train', per)
##
regr = LinearRegression().fit(trn.X, trn.y)
##
print('Test of Lemmas Model')
tst = STSModels()
tst.fit_lemmas(filename, 'test', per)
##
print('Prediction')
pred = regr.predict(tst.X)

result = pearsonr(pred, tst.y)[0] # taking just the Pearson coeficient
print(result)
#print(tst.y - pred)
