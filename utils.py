import random
import json
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

def random_split(data,batch_size=50):
    if len(data) <= 50:
        return [data]
    res = []
    random.shuffle(data)
    cur = []
    for d in data:
        if len(cur) < batch_size:
            cur.append(d)
        else:
            res.append(cur)
            cur = [d]
    if cur != []:
        res.append(cur)
    return res

def build_word_vec():
    all_word_vec = dict()
    with open("./data/glove.6B.300d.txt","r") as rf:
        for line in rf:
            content = line.strip().split()
            word = content[0]
            vec = [float(x) for x in content[1:]]
            all_word_vec[word] = vec
    mdl_data = json.load(open("./data/mdl_data.json","r"))
    res = dict()
    for i,adj in enumerate(mdl_data):
        if i % 1000 == 0 and i != 0:
            print(i)
        nouns = list(mdl_data[adj].keys())
        for noun in nouns:
            if noun in all_word_vec:
                res[noun] = all_word_vec[noun]
    json.dump(res,open("./data/word_vec.json","w"))
    return res

def visualise_cluster(data,labels):
    tsne = TSNE(n_components=2)
    d_data = tsne.fit_transform(data)
    x = []
    y = []
    for d in d_data:
        x.append(d[0])
        y.append(d[1])
    plt.scatter(x,y,c=labels, marker="x")  
    plt.savefig("./result/cluster_res.pdf") 
    plt.close() 

def dbscan(data,precomputed=True,eps=0.96,min_samples=20):
    if precomputed:
        model = DBSCAN(eps=eps,min_samples=min_samples,metric="precomputed")
    else:
        model = DBSCAN(eps=eps,min_samples=min_samples)
    y_pred = model.fit_predict(data)
    all_y = list(y_pred)
    # all_c = list(set(all_y))
    # cur.append(len(all_c))
    # cur.append(sum([1 for y in all_y if y==-1])/len(all_y))
    # record[i].append(cur)
    return [int(x) for x in all_y]