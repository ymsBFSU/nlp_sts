
# coding: utf-8

# In[32]:


import nltk
nltk.download('treebank')
from nltk.corpus import treebank
from nltk.corpus import wordnet as wn
from nltk.tag.perceptron import PerceptronTagger
from nltk import pos_tag
from nltk.wsd import lesk
from nltk.metrics import jaccard_distance
from scipy.stats import pearsonr
from nltk.stem import WordNetLemmatizer
from itertools import chain
from nltk.cluster.util import cosine_distance

#ploting libraries
import matplotlib.pyplot as plt


# In[9]:


# In this case it is not necessary to use a perceptron tagger because it performs almost the same way
train_data = treebank.tagged_sents()
    
#Perceptron tagger
PER = PerceptronTagger(load='false')
PER.train(train_data)


# In[10]:


wn_postag_map = {'N': 'n',
                 'V': 'v',
                 'J': 'a',
                 'R': 'r'
                }


# In[50]:


def new_lesk(context_sentence, ambiguous_word, pos=None, synsets=None):
    context = set(context_sentence)
    hyper = lambda s: s.hypernyms()
    if synsets is None:
        synsets = wn.synsets(ambiguous_word)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None
    
    max_sense = []
    for ss in synsets:
        ss_lst = list(ss.closure(hyper))
        ss_lst.append(ss)
        for ss_hyper in ss_lst:
            max_sense.append(
                (len(context.intersection(ss_hyper.definition().split())), ss_hyper, ss) 
                  )
            for ex in ss_hyper.examples():
                if ex: 
                    max_sense.append(
                        (len(context.intersection(ex.split())), ss_hyper, ss) 
                                    )
    _, hyper, sense = max(max_sense)
    

    return sense


# In[85]:


wnl = WordNetLemmatizer()

def lemmatize(p):
    if p[1][0] in {'N','V'}:
        return wnl.lemmatize(p[0].lower(), pos=p[1][0].lower())
    return p[0].lower()


# In[90]:


def my_WSD(pt_pair, context):
    '''
    Returns the synset if it's a noun, a verb, an adberv or an adjective or the lowered word 
    '''
    pt = wn_postag_map.get(pt_pair[1][0])
    word = pt_pair[0]
    if pt:
        return new_lesk(context, word, pt)
    else:
        return pt_pair[0].lower()


# In[91]:


trainfile_prefix = 'train/STS'
input_prefix = trainfile_prefix + '.input.'
gs_prefix = trainfile_prefix + '.gs.'


# In[92]:


filename = 'MSRpar'


# In[93]:


jd_list = []
js_list = []
for line in open(input_prefix+filename+'.txt','r', encoding='utf-8'):
    # split in sentences using the information given about the structure of the input file
    sent_tab = line.split('\t')
    sent1 = sent_tab[0].strip()
    sent2 = sent_tab[1].strip('\n').strip()

    words1, words2 = [nltk.word_tokenize(s) for s in [sent1,sent2]]
    
    pairs1 = PER.tag(words1) # pos_tag(words1)
    pairs2 = PER.tag(words2) # pos_tag(words2)

    lem_words1 = [lemmatize(pair) for pair in pairs1]
    lem_words2 = [lemmatize(pair) for pair in pairs2]
    
    synsets1 = [my_WSD((lem,pt_pair[1]), lem_words1) for lem, pt_pair in zip(lem_words1, pairs1)]
    synsets1 = [my_WSD((lem,pt_pair[1]), lem_words1) for lem, pt_pair in zip(lem_words2, pairs2)]
    #synsets1 = [my_WSD(pair,words1) for pair in pairs1]
    #synsets2 = [my_WSD(pair,words2) for pair in pairs2]
    
    jd = jaccard_distance(set(synsets1), set(synsets2))
    jd_list.append(jd)
    js_list.append(1-jd)


# In[73]:


gs_list = []
for line in open(gs_prefix+filename+'.txt','r'):
    gs_list.append(int(line.split('\t')[0][0]))


# In[94]:


pearsonr(gs_list, js_list)[0] # taking just the Pearson coeficient

