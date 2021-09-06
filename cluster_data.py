import os
import json
import random
import copy
import numpy as np
from multiprocessing import Pool

from mdl.probase_mdl import CT
from utils import dbscan

def jaccard_distance(n1,n2):
    s = list(set(n1) & set(n2))
    d = len(s)/(len(n1)+len(n2)-len(s)+1e-24)
    return 1-d

def build_jac_matrix(hype=True,hypo=False):
    mdl_data = json.load(open("./data/mdl_prob_data.json","r"))
    adjs = ["beautiful","cute","dangerous","expensive","famous","poor","popular","strong","successful","traditional"]
    concept_tree = CT.concept_tree
    
    for i,adj in enumerate(mdl_data):
    # for adj in adjs:
        if i % 1000 == 0 and i != 0:
            print(i)
        if adj in adjs:continue
        print(adj)
        if not os.path.exists("./test_data/adjss/{}".format(adj)):
            os.makedirs("./test_data/adjss/{}".format(adj))
        else:
            continue
        nouns = list(mdl_data[adj].keys())
        json.dump(nouns,open("./test_data/adjss/{}/{}.json".format(adj,adj),"w"))
        X = np.zeros((len(nouns),len(nouns)),dtype="float32")
        noun_set = dict()
        for n in nouns:
            noun_set[n] = [list(concept_tree[n]["children"].keys()),list(concept_tree[n]["parents"].keys())]
        for i,n1 in enumerate(nouns):             
            n1_list = noun_set[n1]
            for j in range(i+1,len(nouns)):
                n2 = nouns[j]
                n2_list = noun_set[n2]
                if hype and hypo:
                    hypo_dis = jaccard_distance(n1_list[0],n2_list[0])
                    hype_dis = jaccard_distance(n1_list[1],n2_list[1])
                    X[i][j] = hypo_dis*hype_dis
                    X[j][i] = hypo_dis*hype_dis
                elif hype:
                    hype_dis = jaccard_distance(n1_list[1],n2_list[1])
                    X[i][j] = hype_dis
                    X[j][i] = hype_dis
                elif hypo:
                    hypo_dis = jaccard_distance(n1_list[0],n2_list[0])
                    X[i][j] = hypo_dis
                    X[j][i] = hypo_dis
                else:
                    print("at least one of hype and hypo must be true")
                    return
        np.save("./test_data/adjss/{}/{}.npy".format(adj,adj),X)

def cluster_iter(thresh=10):
    adjs = ["beautiful","cute","dangerous","expensive","famous","poor","popular","strong","successful","traditional"]
    # for d in os.listdir("./test_data/adjss"):
    #     dn = "./test_data/adjss/"+d
    #     if not os.path.isdir(dn):continue 
    #     adj = d
    #     if adj in adjs:continue
    for adj in adjs:
        print(adj)
        data = np.load("./test_data/adjs/{}/{}.npy".format(adj,adj))
        test_data = json.load(open("./test_data/adjs/{}/{}.json".format(adj,adj),"r"))
        noun2id = {noun:i for i,noun in enumerate(test_data)}

        result = [[],[],[]]
        cur_data = [copy.deepcopy(data),copy.deepcopy(data),copy.deepcopy(data)]
        cur_noun = [copy.deepcopy(test_data),copy.deepcopy(test_data),copy.deepcopy(test_data)]
        last_noun = [len(cur_noun[0]),len(cur_noun[1]),len(cur_noun[2])]

        iter_count = 1
        eps = [0.9,0.9,0.9]
        min_samples = [3,3,3]
        end_flag = [0,0,0]
        type_dict = {0:"all nouns",1:"normal nouns",2:"outlier nouns"}
        while True:
            if sum(end_flag) == 3: break
            print("\n第%d次聚类迭代：" % iter_count)
            iter_count += 1
            for i in range(3):
                if end_flag[i]: continue
                print("Clustering using {}: ".format(type_dict[i]))
                labels = dbscan(cur_data[i],precomputed=True,eps=eps[i],min_samples=min_samples[i])
                res = dict()
                for k,c in enumerate(labels):
                    if c not in res:
                        res[c] = [cur_noun[i][k]]
                    else:
                        res[c].append(cur_noun[i][k])       
                left_words = []
                for c in res:
                    if len(res[c]) < thresh and c != -1:
                        result[i].append(res[c])
                    elif i == 0:
                        left_words.extend(res[c])
                    elif i == 1 and c != -1:
                        left_words.extend(res[c])
                    elif i == 2 and c == -1:
                        left_words = res[c]

                if left_words != []:
                    X = np.zeros((len(left_words),len(left_words)),dtype="float32")
                    for k,n1 in enumerate(left_words):
                        for j in range(k+1,len(left_words)):
                            n2 = left_words[j]
                            X[k][j] = data[noun2id[n1]][noun2id[n2]]
                            X[j][k] = data[noun2id[n1]][noun2id[n2]]

                    cur_data[i] = X
                    cur_noun[i] = left_words

                    if len(left_words) == last_noun[i]:
                        if i == 1:
                            eps[i] = max(0.85,eps[i]-0.005)
                            if eps[i] == 0.85:
                                end_flag[i] = 1
                                print("Clustering using {} has been ended!".format(type_dict[i]))
                        else:
                            # min_samples[i] = max(1,min_samples[i]-1)
                            eps[i] = min(0.92,eps[i]+0.005)
                            if eps[i] == 0.92:
                            # if min_samples[i] == 1 and eps[i] == 0.99:
                                end_flag[i] = 1
                                print("Clustering using {} has been ended!".format(type_dict[i]))
                        print("min_samples: %d" % min_samples[i])
                        print("eps: %f" % eps[i])
                    last_noun[i] = len(left_words)

                else:
                    end_flag[i] = 1
                    print("Clustering using {} has been ended!".format(type_dict[i]))
        if not os.path.exists("./result/adjss/{}".format(adj)):
            os.makedirs("./result/adjss/{}".format(adj))
        with open("./result/adjss/{}/base_iter_result.txt".format(adj),"w") as wf:
            for r in result[0]:
                wf.write(",".join(r)+"\n")
        with open("./result/adjss/{}/normal_iter_result.txt".format(adj),"w") as wf:
            for r in result[1]:
                wf.write(",".join(r)+"\n")
        with open("./result/adjss/{}/abnormal_iter_result.txt".format(adj),"w") as wf:
            for r in result[2]:
                wf.write(",".join(r)+"\n")

def cluster_by_neighbor(K=5):
    adjs = ["beautiful"]
    for adj in adjs:
        print(adj)
        res = []
        nouns = json.load(open("./test_data/{}/{}.json".format(adj,adj),"r"))
        dis = np.load("./test_data/{}/{}_hype.npy".format(adj,adj))
        for i,n in enumerate(nouns):
            inx = dis[i].argsort()[1:K+1]
            res.append([n] + [nouns[j] for j in inx])
        with open("./result/{}/neighbor.txt".format(adj),"w") as wf:
            wf.write("\n".join([",".join(x) for x in res]))

def matrix_process(name,test_adjs):
    print("runing process %s" % name)
    hype = True
    hypo = False
    for i,adj in enumerate(test_adjs):
        if i % 1000 == 0 and i != 0:
            print(name,i)
        if adj in adjs:continue
        if not os.path.exists("./test_data/adjss/{}".format(adj)):
            os.makedirs("./test_data/adjss/{}".format(adj))
        else:
            continue
        print(name,adj)
        nouns = list(mdl_data[adj].keys())
        json.dump(nouns,open("./test_data/adjss/{}/{}.json".format(adj,adj),"w"))
        X = np.zeros((len(nouns),len(nouns)),dtype="float32")
        noun_set = dict()
        for n in nouns:
            # if not n in noun_set:
            noun_set[n] = [list(concept_tree[n]["children"].keys()),list(concept_tree[n]["parents"].keys())]
        for i,n1 in enumerate(nouns):             
            n1_list = noun_set[n1]
            for j in range(i+1,len(nouns)):
                n2 = nouns[j]
                # nk = n1+"##"+n2 if n1 < n2 else n2+"##"+n1
                # if nk in dist_dict:
                #     X[i][j] = dist_dict[nk]
                #     X[j][i] = dist_dict[nk]
                #     continue

                n2_list = noun_set[n2]
                if hype and hypo:
                    hypo_dis = jaccard_distance(n1_list[0],n2_list[0])
                    hype_dis = jaccard_distance(n1_list[1],n2_list[1])
                    X[i][j] = hypo_dis*hype_dis
                    X[j][i] = hypo_dis*hype_dis
                    # dist_dict[nk] = hypo_dis*hype_dis
                elif hype:
                    hype_dis = jaccard_distance(n1_list[1],n2_list[1])
                    X[i][j] = hype_dis
                    X[j][i] = hype_dis
                    # dist_dict[nk] = hype_dis
                elif hypo:
                    hypo_dis = jaccard_distance(n1_list[0],n2_list[0])
                    X[i][j] = hypo_dis
                    X[j][i] = hypo_dis
                    # dist_dict[nk] = hypo_dis
                else:
                    print("at least one of hype and hypo must be true")
                    return
        np.save("./test_data/adjss/{}/{}.npy".format(adj,adj),X)
    
    print("process %s has been done" % name)

def cluster_process(name,test_adjs,thresh=10):
    for adj in test_adjs:
        dn = "./test_data/adjss/"+adj
        if not os.path.isdir(dn):continue 
        if adj in adjs:continue
        if not os.path.exists("./result/adjss/{}".format(adj)):
            os.makedirs("./result/adjss/{}".format(adj))
        else:
            continue
        print(name,adj)
        
        data = np.load("./test_data/adjss/{}/{}.npy".format(adj,adj))
        test_data = json.load(open("./test_data/adjss/{}/{}.json".format(adj,adj),"r"))
        noun2id = {noun:i for i,noun in enumerate(test_data)}

        result = [[],[],[]]
        cur_data = [copy.deepcopy(data),copy.deepcopy(data),copy.deepcopy(data)]
        cur_noun = [copy.deepcopy(test_data),copy.deepcopy(test_data),copy.deepcopy(test_data)]
        last_noun = [len(cur_noun[0]),len(cur_noun[1]),len(cur_noun[2])]

        iter_count = 1
        eps = [0.9,0.9,0.9]
        min_samples = [3,3,3]
        end_flag = [0,0,0]
        type_dict = {0:"all nouns",1:"normal nouns",2:"outlier nouns"}
        while True:
            if sum(end_flag) == 3: break
            # print("\n第%d次聚类迭代：" % iter_count)
            iter_count += 1
            for i in range(3):
                if end_flag[i]: continue
                # print("Clustering using {}: ".format(type_dict[i]))
                labels = dbscan(cur_data[i],precomputed=True,eps=eps[i],min_samples=min_samples[i])
                res = dict()
                for k,c in enumerate(labels):
                    if c not in res:
                        res[c] = [cur_noun[i][k]]
                    else:
                        res[c].append(cur_noun[i][k])       
                left_words = []
                for c in res:
                    if len(res[c]) < thresh and c != -1:
                        result[i].append(res[c])
                    elif i == 0:
                        left_words.extend(res[c])
                    elif i == 1 and c != -1:
                        left_words.extend(res[c])
                    elif i == 2 and c == -1:
                        left_words = res[c]

                if left_words != []:
                    X = np.zeros((len(left_words),len(left_words)),dtype="float32")
                    for k,n1 in enumerate(left_words):
                        for j in range(k+1,len(left_words)):
                            n2 = left_words[j]
                            X[k][j] = data[noun2id[n1]][noun2id[n2]]
                            X[j][k] = data[noun2id[n1]][noun2id[n2]]

                    cur_data[i] = X
                    cur_noun[i] = left_words

                    if len(left_words) == last_noun[i]:
                        if i == 1:
                            eps[i] = max(0.85,eps[i]-0.005)
                            if eps[i] == 0.85:
                                end_flag[i] = 1
                                # print("Clustering using {} has been ended!".format(type_dict[i]))
                        else:
                            # min_samples[i] = max(1,min_samples[i]-1)
                            eps[i] = min(0.92,eps[i]+0.005)
                            if eps[i] == 0.92:
                            # if min_samples[i] == 1 and eps[i] == 0.99:
                                end_flag[i] = 1
                                # print("Clustering using {} has been ended!".format(type_dict[i]))
                        # print("min_samples: %d" % min_samples[i])
                        # print("eps: %f" % eps[i])
                    last_noun[i] = len(left_words)

                else:
                    end_flag[i] = 1
                    # print("Clustering using {} has been ended!".format(type_dict[i]))

        with open("./result/adjss/{}/base_iter_result.txt".format(adj),"w") as wf:
            for r in result[0]:
                wf.write(",".join(r)+"\n")
        with open("./result/adjss/{}/normal_iter_result.txt".format(adj),"w") as wf:
            for r in result[1]:
                wf.write(",".join(r)+"\n")
        with open("./result/adjss/{}/abnormal_iter_result.txt".format(adj),"w") as wf:
            for r in result[2]:
                wf.write(",".join(r)+"\n")

mdl_data = json.load(open("./data/mdl_prob_data.json","r"))
adjs = ["beautiful","cute","dangerous","expensive","famous","poor","popular","strong","successful","traditional"]
concept_tree = CT.concept_tree
noun_set = {}
dist_dict = {}

def matrix_main():
    """
    Function:
        multi-process for building jacard distance between nouns
    Output:
        *.npy for all adjectives saving jacard distance between nouns
    """
    print("parent process %s" % os.getpid())
    all_adjs = list(mdl_data.keys())
    step = int(len(all_adjs)/4) + 1

    # all_adjs = all_adjs[:step * 3]
    # random.shuffle(all_adjs)
    # step = int(len(all_adjs)/3) + 1

    p = Pool(4)
    for i in range(4):
        p.apply_async(matrix_process,args=(str(i),all_adjs[i * step:min((i+1)*step,len(all_adjs))]))
    p.close()
    p.join()
    print("all proceess have been done")

def cluster_main():
    """
    Function:
        multi-process for clustering nouns
    Output:
        */normal_iter_result.txt saving clustering results
    """
    all_adjs = []
    clustered_adjs = set()
    with open("./clustered.txt","r") as rf:
        for line in rf:
            clustered_adjs.add(line.strip())
    for d in os.listdir("./test_data/adjss"):
        dn = os.path.join("./test_data/adjss",d)
        if not os.path.isdir(dn):
            continue
        if not d in clustered_adjs:
            all_adjs.append(d)
    print("parent process %s" % os.getpid())
    print(len(all_adjs))
    # all_adjs = all_adjs[int(len(all_adjs)/2):]
    step = int(len(all_adjs)/8) + 1

    p = Pool(8)
    for i in range(8):
        p.apply_async(cluster_process,args=(str(i),all_adjs[i * step:min((i+1)*step,len(all_adjs))]))
    p.close()
    p.join()
    print("all proceess have been done")
    
if __name__ == '__main__':
    matrix_main()
    cluster_main()