##
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.metrics import jaccard_distance
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

stop_words = stopwords.words('english')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
wnl = WordNetLemmatizer()
filenames_train = ['MSRpar', 'MSRvid', 'SMTeuroparl']
filenames_test = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                  'surprise.OnWN', 'surprise.SMTnews']


file_prefix = lambda stage: stage + '/STS'

""" more orthodox and robust implementation """
def dice_coefficient(s1, s2, n=2):
    """dice coefficient 2nt/na + nb."""
    if isinstance((s1), list):
        a = ' '.join(s1)
    elif isinstance(s1, str):
        a = s1
    if isinstance((s2), list):
        b = ' '.join(s2)
    elif isinstance(s2, str):
        b = s2
    if not len(a) or not len(b): return 0.0
    if len(a) == 1:  a = a + u'.'
    if len(b) == 1:  b = b + u'.'

    a_bigram_list = []
    for i in range(len(a) - 1):
        a_bigram_list.append(a[i:i + n])
    b_bigram_list = []
    for i in range(len(b) - 1):
        b_bigram_list.append(b[i:i + n])

    a_bigrams = set(a_bigram_list)
    b_bigrams = set(b_bigram_list)
    overlap = len(a_bigrams & b_bigrams)
    dice_coeff = overlap * 2.0 / (len(a_bigrams) + len(b_bigrams))
    return dice_coeff

def getData(filenames, file_prefix, **removals):
    input_prefix = file_prefix + '.input.'
    gs_prefix = file_prefix + '.gs.'
    input = pd.DataFrame()
    for filename in filenames:
        sentences = pd.read_csv(input_prefix + filename + '.txt',
                                sep='\t', names=['sentence1', 'sentence2'],
                                quoting=3)
        golden_standards = pd.read_csv(gs_prefix + filename + '.txt',
                                       names=['golden_standard'])
        dfX = pd.concat([sentences, golden_standards], axis=1)
        input = pd.concat([input, dfX])

    input.reset_index(drop=True)
    # Remove punctuation (and numbers)
    input[['sentence1', 'sentence2']] = input[['sentence1', 'sentence2']].apply(lambda col: sentClean(col, **removals))
    # Tokenize (remove stop words)
    input[['sentence1', 'sentence2']] = input[['sentence1', 'sentence2']].apply(lambda col: sentTokenize(col, **removals))
    return input


def sentClean(sents, **removals):
    new_sent = []
    for sent in sents:
        if sent is np.NaN:
            pass#print(sent)
        if removals.get('numbers'):
            mod_sent = re.sub(r'[^A-Z a-z]', ' ', sent)
        else:
            mod_sent = re.sub(r'[^\w]', ' ', sent)  # |\b\w\b -> to take out single chars
        clean_sent = re.sub(r'[ ]+', ' ', mod_sent.strip())
        if len(clean_sent) == 0:
            clean_sent = sent
        new_sent.append(clean_sent)
    return new_sent

def sentTokenize(sents, **removals):
    new_sent = []
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        if removals.get('stop_words'):
            tokens = [token for token in tokens if token not in stop_words]
        new_sent.append(tokens)
    return new_sent

def lemmatize(p):
    if p[1][0] in {'N','V'}:
        return wnl.lemmatize(p[0].lower(), pos=p[1][0].lower())
    return p[0].lower()

def lemmas(sents):
    new_sent = []
    for sent in sents:
        pairs = pos_tag(sent)
        new_sent.append([lemmatize(pair) for pair in pairs])
    return new_sent

def ngrams(sents, n, word=True):
    new_sent = []
    n_old = n
    for sent in sents:
        n = n_old
        if len(sent) < n:
            n_old = n
            n = len(sent)
        if word:
            grams_lst = [w for w in nltk.ngrams(sent, n)]
            new_sent.append(grams_lst)
        else:
            sent_joined = ' '.join(sent)
            grams_lst = [''.join(ch) for ch in nltk.ngrams(sent_joined, n)]
            new_sent.append(grams_lst)
    return new_sent

def mylesk(context_sentence, ambiguous_word, pos=None, synsets=None):
    context = set(context_sentence)
    # context = ' '.join(context_sentence)
    hyper = lambda s: s.hypernyms()
    if synsets is None:
        synsets = wn.synsets(ambiguous_word)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return ambiguous_word

    max_sense = []
    for ss in synsets:
        ss_lst = list(ss.closure(hyper))
        ss_lst.append(ss)
        for ss_hyper in ss_lst:
            max_sense.append(
                (len(context.intersection(ss_hyper.definition().split())), ss_hyper, ss)
                # (self.dice_coefficient(context, ss_hyper.definition()), ss_hyper, ss)
            )
            for ex in ss_hyper.examples():
                if ex:
                    max_sense.append(
                        (len(context.intersection(ex.split())), ss_hyper, ss)
                        # (self.dice_coefficient(context, ex), ss_hyper, ss)
                    )
    _, hyper, sense = max(max_sense)
    return hyper.lemmas()[0].name()

def myWSD(pt_pair, context):
    '''
    Returns the synset if it's a noun, a verb, an adberv or an adjective or the lowered word
    '''
    wn_postag_map = {'N': 'n',
             'V': 'v',
             'J': 'a',
             'R': 'r'
            }

    pt = wn_postag_map.get(pt_pair[1][0])
    word = pt_pair[0]
    if pt:
        return mylesk(context, word, pt)
    else:
        return word

def lesk(sents):
    new_sent = []
    for sent in sents:
        pairs = pos_tag(sent)
        new_sent.append([myWSD(pair,sent) for pair in pairs])
    return new_sent

def modelFeatures(df_sents):
    Xlem = df_sents.apply(lemmas)
    Xlesk = Xlem.apply(lesk)

    Mdice_lem = Xlem.apply(lambda x: dice_coefficient(x['sentence1'], x['sentence2']), axis=1)

    Mjac_lem = Xlem.apply(lambda x: jaccard_distance(set(x['sentence1']), set(x['sentence2'])), axis=1)

    #Mlesk = Xlesk.apply(lambda x: jaccard_distance(set(x['sentence1']), set(x['sentence2'])), axis=1)
    Mlesk = Xlesk.apply(lambda x: dice_coefficient(x['sentence1'], x['sentence2']), axis=1)
    removals = {
        'stop_words': True,
        'numbers': True
    }
    Xngram = Xlem.apply(lambda col: ngrams(col, 2))
    Mngram2 = Xngram.apply(lambda x: jaccard_distance(set(x['sentence1']), set(x['sentence2'])), axis=1)

    Xngram = Xlem.apply(lambda col: ngrams(col, 4))
    Mngram4 = Xngram.apply(lambda x: jaccard_distance(set(x['sentence1']), set(x['sentence2'])), axis=1)
    '''
    df_XtrainSW = getData(filenames_train, file_prefix('train'), **removals)
    Xlem = df_XtrainSW.apply(lemmas)
    Xngram = Xlem.apply(lambda col: ngrams(col, 1))
    Mngram1_sw = Xngram.apply(lambda x: jaccard_distance(set(x['sentence1']), set(x['sentence2'])), axis=1)
    Xngram = Xlem.apply(lambda col: ngrams(col, 2))
    Mngram2_sw = Xngram.apply(lambda x: jaccard_distance(set(x['sentence1']), set(x['sentence2'])), axis=1)
    
    df = pd.concat([Mdice_lem, Mjac_lem, Mlesk, Mngram2, Mngram4, Mngram1_sw, Mngram2_sw], axis=1)
    '''
    df = pd.concat([Mdice_lem, Mjac_lem, Mlesk, Mngram2, Mngram4], axis=1)
    return df

#input_prefix = lambda stage:  file_prefix(stage) + '.input.'
#gs_prefix = lambda stage: file_prefix(stage) + '.gs.'

removals = {
            'stop_words': False,
            'numbers': True
            }
# Tokenized raw train data
df_Xtrain = getData(filenames_train, file_prefix('train'), **removals)
#df_Xtrain.sample(frac=1)
labels_trn = df_Xtrain.iloc[:, -1]
df_Xtrain = df_Xtrain.drop(df_Xtrain.columns[len(df_Xtrain.columns) - 1], axis=1)

# Tokenized raw test data
df_Xtest = getData(filenames_test, file_prefix('test'), **removals)
labels_tst = df_Xtest.iloc[:, -1]
df_Xtest = df_Xtest.drop(df_Xtest.columns[len(df_Xtest.columns) - 1], axis=1)

'''
Xtrn, Xval, ytrn, yval = train_test_split(
                                        df_Xtrain[['sentence1', 'sentence2']],
                                        df_Xtrain['golden_standard'],
                                        test_size=0.4)
                                        #random_state=0)
'''
##
#print(Xlem.head())
#Xngram = Xlem.apply(lambda col: ngrams(col, n))

transformed_train = modelFeatures(df_Xtrain)
reg = LinearRegression().fit(transformed_train, labels_trn)

transformed_test = modelFeatures(df_Xtest)
pred = reg.predict(transformed_test)
print(pearsonr(pred.T.tolist(), labels_tst.tolist())[0])