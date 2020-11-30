import os
import math
import numpy as np
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.Session(config=conf)
set_session(sess)

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import uniout

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"bert/models/uncased_L-12_H-768_A-12")

config_path = os.path.join(MODEL_DIR,"bert_config.json")
checkpoint_path = os.path.join(MODEL_DIR,"bert_model.ckpt")
dict_path = os.path.join(MODEL_DIR,"vocab.txt")

graph = tf.get_default_graph()
tokenizer = Tokenizer(dict_path, do_lower_case=True)
with graph.as_default():
    model = build_transformer_model(config_path, checkpoint_path)

def euclid_distance(x,y):
    return np.sqrt(((x-y) ** 2).sum())

def cosine_distance(x,y):
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    sim = np.dot(x,y.T)/(x_norm * y_norm)
    return 1.-sim

def match_tokenized_to_untokenize(subwords,adj,noun):
    adj_words = adj.split()
    noun_words = noun.split()
    
    in_adj = True
    adj_index = 0
    noun_index = 0
    token_ids = ["O"] * len(subwords)
    last = ""
    for i,subword in enumerate(subwords):
        if subword in ["[CLS]","[SEP]"]:
            continue

        if subword.startswith("##"):
            last += subword[2:]
        else:
            last += subword

        if in_adj:
            token_ids[i] = "A-{}".format(str(adj_index))
            if last == adj_words[0]:
                adj_words.remove(last)
                adj_index += 1
                last = ""
                if len(adj_words) == 0:
                    in_adj = False
        else:
            token_ids[i] = "N-{}".format(str(noun_index))
            if last == noun_words[0]:
                noun_words.remove(last)
                noun_index += 1
                last = ""

    return token_ids

def adj_pmi(adj,noun,dist="euc"):
    text = adj + " " + noun
    token_ids,segment_ids = tokenizer.encode(text)
    tokenized_text = tokenizer.tokenize(text)
    mapping = match_tokenized_to_untokenize(tokenized_text,adj,noun)
    # print(token_ids)
    # print(tokenized_text)
    # print(mapping)

    length = 3
    batch_token_ids = np.array([token_ids] * 3)
    batch_segment_ids = np.zeros_like(batch_token_ids)
    for i in range(length):
        if i == 0:
            for j in range(len(mapping)):
                if mapping[j].startswith("A"):
                    batch_token_ids[i,j] = tokenizer._token_mask_id
        elif i == 1:
            for j in range(len(mapping)):
                if mapping[j] != "O":
                    batch_token_ids[i,j] = tokenizer._token_mask_id
        else:
            for j in range(len(mapping)):
                if mapping[j].startswith("N"):
                    batch_token_ids[i,j] = tokenizer._token_mask_id
    with graph.as_default():
        vectors = model.predict([batch_token_ids,batch_segment_ids])
    if dist == "euc":
        dist_f = euclid_distance
    else:
        dist_f = cosine_distance
    vec = []
    for i in range(length):
        vec_dict = {}
        for j in range(len(mapping)):
            if mapping[j] == "O":
                continue
            if not mapping[j] in vec_dict:
                vec_dict[mapping[j]] = [vectors[i,j]]
            else:
                vec_dict[mapping[j]].append(vectors[i,j])
        adj_vec = []
        noun_vec = []
        for k in vec_dict:
            v = np.mean(vec_dict[k],axis=0)
            if k.startswith("A"):
                adj_vec.append(v)
            else:
                noun_vec.append(v)
        vec.append([np.mean(adj_vec,axis=0),np.mean(noun_vec,axis=0)])

    d1 = dist_f(vec[0][0],vec[1][0])
    d2 = dist_f(vec[2][1],vec[1][1])
    d = (d1+d2)/2
    d = 1./math.log(d)

    return d


def bert_pmi(texts,dist="euc"):
    res = []
    for text in texts:
        adj,noun = text
        res.append(adj_pmi(adj,noun,dist=dist))
    return res

if __name__ == '__main__':
    adj = "beautiful"
    noun = "bird"
    print(adj_pmi(adj,noun))
    noun = "common bird"
    print(adj_pmi(adj,noun))

    adj = "beautiful"
    noun = "fabric"
    print(adj_pmi(adj,noun))
    noun = "silk fabric"
    print(adj_pmi(adj,noun))

    adj = "successful"
    noun = "summaries"

    adj_pmi(adj,noun)