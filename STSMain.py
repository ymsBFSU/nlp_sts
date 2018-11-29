import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

stop_words = stopwords.words('english')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
wnl = WordNetLemmatizer()

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
        new_sent.append(re.sub(r'[ ]+', ' ', mod_sent.strip()))
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
    for sent in sents:
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
    return sense

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

filenames_train = ['MSRpar', 'MSRvid', 'SMTeuroparl']
filenames_test = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                  'surprise.OnWN', 'surprise.SMTnews']


file_prefix = lambda stage: stage + '/STS'
#input_prefix = lambda stage:  file_prefix(stage) + '.input.'
#gs_prefix = lambda stage: file_prefix(stage) + '.gs.'

removals = {
            'stop_words': False,
            'numbers': True
            }
# Tokenized raw train data
df_Xtrain = getData(filenames_train, file_prefix('train'), **removals)
df_Xtrain.sample(frac=1)

# Tokenized raw test data
#df_Xtest = getData(filenames_test, file_prefix('test'), **removals)

Xtrn, Xval, ytrn, yval = train_test_split(
                                        df_Xtrain[['sentence1', 'sentence2']],
                                        df_Xtrain['golden_standard'],
                                        test_size=0.4)
                                        #random_state=0)
Xlem = Xtrn.apply(lemmas)
print(Xlem.head())
n = 4
#Xngram = Xlem.apply(lambda col: ngrams(col, n))

Xlesk = Xlem.apply(lesk)
print(Xlesk.head())
