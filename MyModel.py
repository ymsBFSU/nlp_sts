import nltk
import re
#from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.metrics import jaccard_distance
from scipy.stats import pearsonr
import numpy as np


class STSModels:
    def lemmatize(self, p, wnl):
        if p[1][0] in {'N','V'}:
            return wnl.lemmatize(p[0].lower(), pos=p[1][0].lower())
        return p[0].lower()

    def fit_lemmas(self, filename, stage, per):
        file_prefix = stage + '/STS'
        input_prefix = file_prefix + '.input.'
        gs_prefix = file_prefix + '.gs.'
        wnl = WordNetLemmatizer()

        sim_lst = []
        for line in open(input_prefix+filename+'.txt','r', encoding='utf-8'):
            # split in sentences using the information given about the structure of the input file
            sent_tab = line.split('\t')
            sent1 = sentClean(sent_tab[0].strip())
            sent2 = sentClean(sent_tab[1].strip('\n').strip())

            words1, words2 = [nltk.word_tokenize(s) for s in [sent1,sent2]]
            
            pairs1 = per.tag(words1) # pos_tag(words1)
            pairs2 = per.tag(words2) # pos_tag(words2)

            lem_words1 = [self.lemmatize(pair, wnl) for pair in pairs1]
            lem_words2 = [self.lemmatize(pair, wnl) for pair in pairs2]
            jd = jaccard_distance(set(lem_words1), set(lem_words2))
            sim_lst.append(1-jd)
        self.X = np.array(sim_lst).reshape(-1,1)

        gs_lst = []
        for line in open(gs_prefix+filename+'.txt','r'):
            gs_lst.append(int(line.split('\t')[0][0]))
        self.y = np.array(gs_lst).reshape(-1,1)
        


def sentClean(sent):
        new_sent_lower = sent.lower()
        new_sent = re.sub(r'[^\w]', ' ', new_sent_lower) # |\b\w\b -> to take out single chars
        new_sent = re.sub(r'[ ]+', ' ', new_sent.strip())
        return sent
