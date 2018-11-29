import nltk
import re
#from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import wordnet as wn
from nltk.metrics import jaccard_distance
from scipy.stats import pearsonr
from nltk import pos_tag

import numpy as np
import pandas as pd
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

class STSModels:
    def __init__(self, fit_model, per=None):
        self.fit_model = fit_model

    def lemmatize(self, p, wnl):
        if p[1][0] in {'N','V'}:
            return wnl.lemmatize(p[0].lower(), pos=p[1][0].lower())
        return p[0].lower()

    def lcs(self, a, b):
        lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
        # row 0 and column 0 are initialized to 0 already
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                if x == y:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
        # read the substring out from the matrix
        result = ""
        x, y = len(a), len(b)
        while x != 0 and y != 0:
            if lengths[x][y] == lengths[x-1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y-1]:
                y -= 1
            else:
                assert a[x-1] == b[y-1]
                result = a[x-1] + result
                x -= 1
                y -= 1
        return result

    """ more orthodox and robust implementation """
    def dice_coefficient(self, a, b, n=2):
        """dice coefficient 2nt/na + nb."""
        if not len(a) or not len(b): return 0.0
        if len(a) == 1:  a=a+u'.'
        if len(b) == 1:  b=b+u'.'
        
        a_bigram_list=[]
        for i in range(len(a)-1):
          a_bigram_list.append(a[i:i+n])
        b_bigram_list=[]
        for i in range(len(b)-1):
          b_bigram_list.append(b[i:i+n])
          
        a_bigrams = set(a_bigram_list)
        b_bigrams = set(b_bigram_list)
        overlap = len(a_bigrams & b_bigrams)
        dice_coeff = overlap * 2.0/(len(a_bigrams) + len(b_bigrams))
        return dice_coeff

    def mylesk(self, context_sentence, ambiguous_word, pos=None, synsets=None):
        context = set(context_sentence)
        #context = ' '.join(context_sentence)
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
                    #(self.dice_coefficient(context, ss_hyper.definition()), ss_hyper, ss)
                      )
                for ex in ss_hyper.examples():
                    if ex: 
                        max_sense.append(
                            (len(context.intersection(ex.split())), ss_hyper, ss)
                            #(self.dice_coefficient(context, ex), ss_hyper, ss)
                                        )
        _, hyper, sense = max(max_sense)
        

        return sense

    def myWSD(self, pt_pair, context):
        '''
        Returns the synset if it's a noun, a verb, an adberv or an adjective or the lowered word 
        '''
        wn_postag_map = {'N': 'n',
                 'V': 'v',
                 'J': 'a',
                 'R': 'r'
                }

        pt = wn_postag_map.get(pt_pair[1][0])
        word = pt_pair[0].lower()
        if pt:
            return self.mylesk(context, word, pt)
        else:
            return word

    def lemmasFromSynset(self, synset_lst, most_common=True):
        extended_sent = []
        for word in synset_lst:
            if not isinstance(word, str):
                if most_common:
                    extended_sent.append(word.lemmas()[0].name())
                else:
                    for lemma in word.lemmas():
                        extended_sent.append(lemma.name())
            else:
                extended_sent.append(word)
        return extended_sent

    def fit_lemmas(self, pairs1, pairs2):
        wnl = WordNetLemmatizer()
        lem_words1 = [self.lemmatize(pair, wnl) for pair in pairs1]
        lem_words2 = [self.lemmatize(pair, wnl) for pair in pairs2]
        #sim = jaccard_distance(set(lem_words1), set(lem_words2))
        sim = self.dice_coefficient(' '.join(lem_words1), ' '.join(lem_words2))
        self.X.append(sim)

    def fit_lesk(self, pairs1, pairs2, words1, words2):
        synsets1 = [self.myWSD(pair,words1) for pair in pairs1]
        synsets2 = [self.myWSD(pair,words2) for pair in pairs2]
        #sim = jaccard_distance(set(synsets1), set(synsets2))
        sim = self.dice_coefficient(' '.join(self.lemmasFromSynset(synsets1)),
            ' '.join(self.lemmasFromSynset(synsets2)))
        #sim = self.lcs(''.join(self.lemmasFromSynset(synsets1, most_common=False)),
        #    ''.join(self.lemmasFromSynset(synsets2, most_common=False)))
        #self.X.append(len(sim))
        self.X.append(sim)

        
    def fit_synsets(self, pairs1, pairs2, words1, words2):
        wn_postag_map = {'N': 'n',
                 'V': 'v',
                 'J': 'a',
                 'R': 'r'
                }
        synsets1 = [wn.synsets(pair[0], wn_postag_map.get(pair[1][0])) for pair in pairs1]
        synsets2 = [wn.synsets(pair[0], wn_postag_map.get(pair[1][0])) for pair in pairs2]
        sim_lst = []

        for synset1 in synsets1:
            sim = []
            #for syn1 in synset1:
            if synset1:
                syn1 = synset1[0]
            else:
                syn1 = ''
            if isinstance(syn1, nltk.corpus.reader.wordnet.Synset):
                for synset2 in synsets2:
                    if synset2:
                        syn2 = synset2[0]
                    else:
                        syn2 = ''
                    #for syn2 in synset2:
                    if isinstance(syn2, nltk.corpus.reader.wordnet.Synset):
                        try:
                            sim.append(syn1.lch_similarity(syn2))
                        except:
                            sim.append(0)
                    else:
                        sim.append(0)
            else:
                for synset2 in synsets2:
                    if synset2:
                        synset2 = synset2[0]
                    if isinstance(synset2, str) and sin1:
                        sim.append(1)
                    else:
                        sim.append(0)
                '''
                sim = map(
                    lambda x: 
                        synset1.lch_similarity(x) if isinstance(x, nltk.corpus.reader.wordnet.Synset) else 0.0,
                    synsets2)
                '''
        sim = list(filter(None, sim))
        if sim:
            sim_lst.append(max(sim))
        else:
            sim_lst.append(0)
        self.X.append(max(sim_lst))

    def fit_grams(self, words1, words2):
        a = ' '.join(words1)
        b = ' '.join(words2)

        for n in range(4):
            a_gram_list = []
            for i in range(len(a)-1):
              a_gram_list.append(a[i:i+n])
            b_gram_list=[]
            for i in range(len(b)-1):
              b_gram_list.append(b[i:i+n])
            sim = jaccard_distance(set(a_gram_list), set(b_gram_list))
            self.X.append(sim)

    def fit_bow(self, filenames, stage):
        file_prefix = stage + '/STS'
        input_prefix = file_prefix + '.input.'
        gs_prefix = file_prefix + '.gs.'
        input = pd.DataFrame()
        for filename in filenames:
            input = pd.concat([input, pd.read_csv(input_prefix+filename+'.txt',
                                                  sep='\t', names=['sentence1', 'sentence2'])])
        print(input.shape)

    def fit(self, filenames, stage, sw=True):
        file_prefix = stage + '/STS'
        input_prefix = file_prefix + '.input.'
        gs_prefix = file_prefix + '.gs.'

        self.X = []
        self.y= []
        for filename in filenames:
            for line in open(input_prefix+filename+'.txt','r', encoding='utf-8'):
                # split in sentences using the information given about the structure of the input file
                sent_tab = line.split('\t')
                sent1 = sentClean(sent_tab[0].strip())
                sent2 = sentClean(sent_tab[1].strip('\n').strip())

                words1, words2 = [nltk.word_tokenize(s) for s in [sent1,sent2]]
                if not sw:
                    words1 = [word for word in words1 if word not in stop_words]
                    words2 = [word for word in words2 if word not in stop_words]

                pairs1 = pos_tag(words1) #per.tag(words1) # pos_tag(words1)
                pairs2 = pos_tag(words2) #per.tag(words2) # pos_tag(words2)

                if 'lemmas' in self.fit_model:
                    self.fit_lemmas(pairs1, pairs2)

                if 'lesk' in self.fit_model:
                    self.fit_lesk(pairs1, pairs2, words1, words2)

                if 'synsets' in self.fit_model:
                    self.fit_synsets(pairs1, pairs2, words1, words2)

                if 'ngrams' in self.fit_model:
                    self.fit_grams(words1, words2)

            for line in open(gs_prefix+filename+'.txt','r'):
                self.y.append(int(line.split('\t')[0][0]))

        if 'bow' in self.fit_model:
            self.fit_bow(filenames, stage)
        if 'ngrams' in self.fit_model:
            self.X = np.array(self.X).reshape(int(len(self.X)/n), n)
        else:
            self.X = np.array(self.X).reshape(-1,1)
        #self.y = np.array(self.y).reshape(-1,1)

    def transform(self, filenames, stage):
        self.fit(filenames, stage)

def sentClean(sent):
        new_sent = re.sub(r'[^A-Z a-z]', ' ', sent) # |\b\w\b -> to take out single chars
        new_sent = re.sub(r'[ ]+', ' ', new_sent.strip())
        return new_sent
