import os
import math
import json
import time
import copy
import numpy as np
from functools import reduce

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

from mdl.probase_mdl import Search,CT
from mdl.probase_mdl import NPMI
from mdl.probase_mdl import Bayes
from semantic.MLM.lm_pmi import bert_pmi
from semantic.bert.run_lm_predict import bert_ppl
# from semantic.gpt_2.gpt_lm import gpt2_ppl

# PPL_MODEL = bert_ppl
PPL_MODEL = bert_pmi
probase = CT.concept_tree

class Cache:
    def __init__(self,):
        self.cache = {}
    
    def get(self,key):
        return self.cache.get(key,None)

    def put(self,key,value):
        self.cache[key] = value

    def empty(self,):
        self.cache = {}

    def __len__(self,):
        return len(self.cache)

cache = Cache()
sample_cache = Cache()
file_cache = Cache()

def weight_sample(nouns,weights,num=10):
    res = []
    if len(nouns) <= num:
        return nouns
    else:
        nw = list(zip(nouns,weights))
        nw = sorted(nw,key=lambda x:x[1],reverse=True)
        nw = nw[:num]
        return [x[0] for x in nw]

class AO(Search):
    """
    Function:
        our main methods for conceptualize nouns and evaluation.
    Output:
        (adjective,concept) pairs.
    """
    def __init__(self,adj,instances,normal_num=1.0,beta=0.5,thresh=120,mode="2P",alpha=0.5,**kwargs):
        self.adj = adj
        self.beta = beta
        self.normal_num = normal_num
        ## precomputed
        self.precomputed_lm_score = dict()
        remove_instance = None
        if "remove_instance" in kwargs:
            remove_instance = kwargs["remove_instance"]
        init_concepts = None
        if "init_concepts" in kwargs:
            init_concepts = kwargs["init_concepts"]
        if "hold_multi" in kwargs:
            hold_multi = kwargs["hold_multi"]
        super(AO,self).__init__(instances,thresh,mode,alpha,
                                remove_instance=remove_instance,
                                init_concepts=init_concepts,
                                hold_multi=hold_multi
                            )

    @staticmethod
    def parse_ppl(ppl_res,sents=None,use_ppl=True):
        res = []
        if use_ppl:
            res = [r["ppl"] for r in ppl_res]
        else:
            for i,sent in enumerate(sents):
                words = sent.split(",")[0].split() + sent.split(",")[1].split()
                r = ppl_res[i]
                pp = {}
                ps = []
                last = ""
                for j,x in enumerate(r["tokens"]):
                    # ps.append(x["prob"])
                    ps.append(-math.log(x["prob"]))
                    if x["token"].startswith("##"):
                        last += x["token"][2:]
                    elif x["token"].startswith("Ġ"):
                        last += x["token"][1:]
                    else:
                        last += x["token"]        
                    if last == words[0]:
                        pp[last] = sum(ps)/len(ps)
                        words.remove(last)
                        last = ""
                        ps = []
                ps = []
                for phrase in sent.split(","):
                    cur_p = []
                    for w in phrase.split():
                        cur_p.append(pp[w])
                    ps.append(sum(cur_p)/len(cur_p))
                # res.append(-math.log(reduce(lambda x,y: x*y,ps))/len(ps)) 
                res.append(sum(ps)/len(ps))
        return res

    def compute_ppl(self,concepts,parse=False):
        ppls = []
        sents = []
        sents_sep = []
        for c in concepts:
            # if c in self.cache:
                # ppls.append(self.cache[c])  
            if cache.get(c):
                ppls.append(cache.get(c))
            else:
                sents.append(self.adj + " "+c if parse else [self.adj,c])
                sents_sep.append(self.adj +","+c)
        if sents != []:
            res = PPL_MODEL(sents)
            ## parse_ppl: several ways to compute ppls
            if parse:
                ppls.extend(self.parse_ppl(res,sents=sents_sep,use_ppl=False))
            else:
                ppls.extend(res)
            for i,c in enumerate(self.concepts):
                # self.cache[c] = ppls[i]
                cache.put(c,ppls[i])
        ## mean value or something else
        return sum(ppls)/len(ppls)

    def compute_score(self,mdl,lm):
        return self.beta * mdl + (1-self.beta) * lm / self.normal_num

    def cut_concepts(self):
        parse = False
        c_index = None
        min_score = float("inf")
        min_i_prob = None
        min_ci_prob = None
        tmp = []
        for c in self.S:
            self.concepts.append(c)
            self.update_concept(c,mode="add")
            if self.mode == "2P":
                i_prob = self.update_concept_2P(c,mode="add")
                score = self.compute_length(i_prob)
                tmp.append((c,score))
                if score < min_score:
                    c_index,min_score,min_i_prob = c,score,i_prob
            else:
                i_prob,ci_prob = self.update_concept_NML(c,mode="add")
                score = self.compute_length(i_prob)
                tmp.append((c,score))
                if score < min_score:
                    c_index,min_score,min_i_prob,min_ci_prob = c,score,i_prob,ci_prob
            self.concepts.remove(c)
            self.c_prob.pop(c)

        tmp = sorted(tmp,key=lambda x:x[1])
        tmp_x = []
        for t in tmp[:self.S_num]:
            self.top_S[t[0]] = t[1]
            tmp_x.append(t[0])
            if cache.get(t[0]):
                continue
            
            if sample_cache.get(t[0]):
                # cs = sample_cache.get(t[0])
                continue

            if parse:
                cs = self.parse_ppl(PPL_MODEL([self.adj+" "+t[0]]),sents=[self.adj+","+t[0]])[0]
            else:
                cs = PPL_MODEL([[self.adj,t[0]]])[0]
            sample_cache.put(t[0],cs)
            snouns = list(probase[t[0]]["children"].keys())
            sweights = list(probase[t[0]]["children"].values())
            sample_ns = weight_sample(snouns,sweights)
            cns = []
            no_cache_ns = []
            for sn in sample_ns:
                if sample_cache.get(sn):
                    cns.append(sample_cache.get(sn))
                else:
                    no_cache_ns.append([sn,self.adj+" "+sn if parse else [self.adj,sn],self.adj+","+sn])
            if parse:
                xx = self.parse_ppl(PPL_MODEL([x[1] for x in no_cache_ns]),sents=[x[2] for x in no_cache_ns])
            else:
                xx = PPL_MODEL([x[1] for x in no_cache_ns])
            cns.extend(xx)
            cns = sum(cns)/len(cns)
            for i in range(len(no_cache_ns)):
                sample_cache.put(no_cache_ns[i][0],xx[i])

            cache.put(t[0],(cns+cs)/2)
            file_cache.put(t[0],{
                "concept_score":cs,
                "instance_score":cns
            })

        self.S = tmp_x
        return c_index,min_score,min_i_prob,min_ci_prob

    def add_concept(self,):
        min_ci_prob = None
        min_score = float("inf")
        for c in self.S:
            self.concepts.append(c)
            self.update_concept(c,mode="add")
            if self.mode == "2P":
                i_prob = self.update_concept_2P(c,mode="add")
                mdl = self.compute_length(i_prob)/len(self.instances)
                ## update
                score = self.compute_score(mdl,self.compute_ppl(self.concepts))
                if score < min_score:
                    c_index,min_score,min_i_prob = c,score,i_prob
            else:
                i_prob,ci_prob = self.update_concept_NML(c,mode="add")
                mdl = self.compute_length(i_prob)/len(self.instances)
                ## update
                score = self.compute_score(mdl,self.compute_ppl(self.concepts))
                if score < min_score:
                    c_index,min_score,min_i_prob,min_ci_prob = c,score,i_prob,ci_prob
            self.concepts.remove(c)
            self.c_prob.pop(c)
        return c_index,min_score,min_i_prob,min_ci_prob 

    def remove_concept(self,):
        if len(self.concepts) <= 1:
            return None,float("inf"),None,None
        min_ci_prob = None
        min_score = float("inf")
        old_concepts = copy.deepcopy(self.concepts)
        for c in old_concepts:
            self.concepts.remove(c)
            self.update_concept(c,mode="remove")
            if self.mode == "2P":
                i_prob = self.update_concept_2P(c,mode="remove")
                mdl = self.compute_length(i_prob)/len(self.instances)
                ## update
                score = self.compute_score(mdl,self.compute_ppl(self.concepts))
                if score < min_score:
                    c_index,min_score,min_i_prob = c,score,i_prob
            else:
                i_prob,ci_prob = self.update_concept_NML(c,mode="remove")
                mdl = self.compute_length(i_prob)
                ## update
                score = self.compute_score(mdl,self.compute_ppl(self.concepts))
                if score < min_score:
                    c_index,min_score,min_i_prob,min_ci_prob = c,score,i_prob,ci_prob
            self.concepts.append(c)
            self.c_prob[c] = self.c_length[c]   
        return c_index,min_score,min_i_prob,min_ci_prob

    def replace_concept(self,):
        min_ci_prob = None
        min_score = float("inf")
        old_concepts = copy.deepcopy(self.concepts)
        for s in self.S:
            for c in old_concepts:
                self.concepts.remove(c)
                self.concepts.append(s)
                self.update_concept(c,mode="relace",new_c=s)
                if self.mode == "2P":
                    i_prob = self.update_concept_2P(c,mode="replace",new_c=s)
                    mdl = self.compute_length(i_prob)
                    ## update
                    score = self.compute_score(mdl,self.compute_ppl(self.concepts))
                    if score < min_score:
                        s_index,c_index,min_score,min_i_prob = s,c,score,i_prob
                else:
                    i_prob,ci_prob = self.update_concept_NML(c,mode="replace",new_c=s)
                    mdl = self.compute_length(i_prob)
                    ## update
                    score = self.compute_score(mdl,self.compute_ppl(self.concepts))
                    if score < min_score:
                        s_index,c_index,min_score,min_i_prob,min_ci_prob = s,c,score,i_prob,ci_prob
                self.concepts.remove(s)
                self.concepts.append(c)
                self.c_prob[c] = self.c_length[c]
                self.c_prob.pop(s)
        return s_index,c_index,min_score,min_i_prob,min_ci_prob

def load_file_cache(filename):
    global cache
    global file_cache
    global sample_cache
    tmp = json.load(open(filename,"r"))
    for c in tmp:
        # cache.put(c,(file_cache[c]["concept_score"]+file_cache["instance_score"])/2.0)
        sample_cache.put(c,tmp[c]["concept_score"])
    
    del tmp

class NPMIAO(NPMI):
    """
    Function:
        NPMI based baselines.
    Output:
        (adjective,concept) pairs.
    """
    def __init__(self,adj,instances,thresh=40,topk=2,normal_num=1.0,beta=0.5,parse=False):
        super(NPMIAO,self).__init__(instances,thresh,topk)
        self.adj = adj
        self.normal_num = normal_num
        self.beta = beta
        self.parse = parse

    @staticmethod
    def parse_ppl(ppl_res,sents=None,use_ppl=True):
        res = []
        if use_ppl:
            res = [r["ppl"] for r in ppl_res]
        else:
            for i,sent in enumerate(sents):
                words = sent.split(",")[0].split() + sent.split(",")[1].split()
                r = ppl_res[i]
                pp = {}
                ps = []
                last = ""
                for j,x in enumerate(r["tokens"]):
                    # ps.append(x["prob"])
                    ps.append(-math.log(x["prob"]))
                    if x["token"].startswith("##"):
                        last += x["token"][2:]
                    elif x["token"].startswith("Ġ"):
                        last += x["token"][1:]
                    else:
                        last += x["token"]        
                    if last == words[0]:
                        pp[last] = sum(ps)/len(ps)
                        words.remove(last)
                        last = ""
                        ps = []
                ps = []
                for phrase in sent.split(","):
                    cur_p = []
                    for w in phrase.split():
                        cur_p.append(pp[w])
                    ps.append(sum(cur_p)/len(cur_p))
                # res.append(-math.log(reduce(lambda x,y: x*y,ps))/len(ps)) 
                t = sum(ps)/len(ps)
                ## for npmi
                res.append(0.133 * t + 0.412)
        return res

    def run(self,):
        for i in self.instances:
            self.i_prob[i] = []
            for p in self.S[i]:
                p_score = self.compute_npmi(p,i)
                if p_score > 0:
                    self.i_prob[i].append([p,p_score])
            self.i_prob[i] = sorted(self.i_prob[i],key=lambda x:x[1],reverse=True)
            self.i_prob[i] = self.i_prob[i][:self.topk]
        
        concepts = set()
        score = []
        for i in self.instances:
            if self.i_prob[i] == []:continue
            tmp = []
            for c_prob in self.i_prob[i]:
                c,p_score = c_prob
                p_score = 1./p_score
                if cache.get(c):
                    l_score = cache.get(c)
                else:
                    if sample_cache.get(c):
                        cs = sample_cache.get(c)
                    else:
                        if self.parse:
                            cs = self.parse_ppl(PPL_MODEL([self.adj+" "+c]),sents=[self.adj+","+c])[0]
                        else:
                            cs = PPL_MODEL([[self.adj,c]])[0]

                    snouns = list(probase[c]["children"].keys())
                    sweights = list(probase[c]["children"].values())
                    sample_ns = weight_sample(snouns,sweights)
                    cns = []
                    no_cache_ns = []
                    for sn in sample_ns:
                        if sample_cache.get(sn):
                            cns.append(sample_cache.get(sn))
                        else:
                            no_cache_ns.append([sn,self.adj+" "+sn if self.parse else [self.adj,sn],self.adj+","+sn])
                    if self.parse:
                        xx = self.parse_ppl(PPL_MODEL([x[1] for x in no_cache_ns]),sents=[x[2] for x in no_cache_ns])
                    else:
                        xx = PPL_MODEL([x[1] for x in no_cache_ns])
                    cns.extend(xx)
                    cns = sum(cns)/len(cns)
                    for i in range(len(no_cache_ns)):
                        sample_cache.put(no_cache_ns[i][0],xx[i])

                    cache.put(c,(cns+cs)/2)
                    file_cache.put(c,{
                        "concept_score":cs,
                        "instance_score":cns
                    }) 
                    l_score = (cns+cs)/2.
                
                tmp.append([c,self.beta * p_score + (1-self.beta) * l_score/self.normal_num])

            tmp = sorted(tmp,key=lambda x:x[1])
            concepts.add(tmp[0][0])
            score.append(tmp[0][1])
            self.i_prob[i] = tmp[0]

        self.concepts = list(concepts)
        if len(score) > 0:
            self.score = sum(score)/len(score)
        else:
            self.score = 0

class BayesAO(Bayes):
    """
    Function:
        Basyes Classification based baselines.
    Output:
        (adjective,concept) pairs.
    """
    def __init__(self,adj,instances,thresh=40,topk=3,normal_num=1.0,beta=0.5,parse=False):
        super(BayesAO,self).__init__(instances,thresh,topk)
        self.adj = adj
        self.normal_num = normal_num 
        self.beta = beta
        self.parse = parse

    @staticmethod
    def parse_ppl(ppl_res,sents=None,use_ppl=True):
        res = []
        if use_ppl:
            res = [r["ppl"] for r in ppl_res]
        else:
            for i,sent in enumerate(sents):
                words = sent.split(",")[0].split() + sent.split(",")[1].split()
                r = ppl_res[i]
                pp = {}
                ps = []
                last = ""
                for j,x in enumerate(r["tokens"]):
                    # ps.append(x["prob"])
                    ps.append(-math.log(x["prob"]))
                    if x["token"].startswith("##"):
                        last += x["token"][2:]
                    elif x["token"].startswith("Ġ"):
                        last += x["token"][1:]
                    else:
                        last += x["token"]        
                    if last == words[0]:
                        pp[last] = sum(ps)/len(ps)
                        words.remove(last)
                        last = ""
                        ps = []
                ps = []
                for phrase in sent.split(","):
                    cur_p = []
                    for w in phrase.split():
                        cur_p.append(pp[w])
                    ps.append(sum(cur_p)/len(cur_p))
                # res.append(-math.log(reduce(lambda x,y: x*y,ps))/len(ps)) 
                t = sum(ps)/len(ps)
                ## for bayes
                res.append(1.046 * t - 2.343)
        return res

    def run(self,):
        if self.S == []:
            return
        tmp = []
        for s in self.S:
            score = CT.get_prob(s)
            for i in self.instances:
                c_score = CT.get_cond_prob(s,i)
                score *= c_score
            score = -math.log(score)/len(self.instances)
            tmp.append((s,score))
        tmp = sorted(tmp,key=lambda x:x[1])

        tmp_2 = []
        for c_score in tmp[:self.topk]:
            c,p_score = c_score
            if cache.get(c):
                l_score = cache.get(c)
            else:
                if sample_cache.get(c):
                    cs = sample_cache.get(c)
                else:
                    if self.parse:
                        cs = self.parse_ppl(PPL_MODEL([self.adj+" "+c]),sents=[self.adj+","+c])[0]
                    else:
                        cs = PPL_MODEL([[self.adj,c]])[0]

                snouns = list(probase[c]["children"].keys())
                sweights = list(probase[c]["children"].values())
                sample_ns = weight_sample(snouns,sweights)
                cns = []
                no_cache_ns = []
                for sn in sample_ns:
                    if sample_cache.get(sn):
                        cns.append(sample_cache.get(sn))
                    else:
                        no_cache_ns.append([sn,self.adj+" "+sn if self.parse else [self.adj,sn],self.adj+","+sn])
                if self.parse:
                    xx = self.parse_ppl(PPL_MODEL([x[1] for x in no_cache_ns]),sents=[x[2] for x in no_cache_ns])
                else:
                    xx = PPL_MODEL([x[1] for x in no_cache_ns])
                cns.extend(xx)
                cns = sum(cns)/len(cns)
                for i in range(len(no_cache_ns)):
                    sample_cache.put(no_cache_ns[i][0],xx[i])

                cache.put(c,(cns+cs)/2)
                file_cache.put(c,{
                    "concept_score":cs,
                    "instance_score":cns
                }) 
                l_score = (cns+cs)/2.
            tmp_2.append([c,self.beta * p_score + (1-self.beta) * l_score*self.normal_num])
        tmp_2 = sorted(tmp_2,key=lambda x:x[1])
        self.concepts = [tmp_2[0][0]]
        self.score = tmp_2[0][1]
        self.i_prob = {i:tmp_2[0] for i in self.instances}

def make_concept(mode="2P",thresh=40,alpha=0.6,normal_num=1.0,beta=0.5,hold_multi=True,init_concepts=True):
    # adjs = ["beautiful","cute","dangerous","expensive","famous","poor","popular","strong","successful","traditional"]
    # for adj in adjs:
    for d in os.listdir("./result/adjss"):
        dn = "./result/adjss/"+d
        if not os.path.isdir(dn):continue 
        adj = d
        
        cache.empty()
        sample_cache.empty()
        file_cache.empty()

        start = time.time()
        print("dealing {} sample:".format(adj))
        dirname = "./result/adjss/{}/".format(adj)
        test_txts = ["normal"]
        for tn in test_txts:
            if os.path.exists(dirname+"{}_sample_pmi.json".format(tn)):
                continue
            textname = dirname + tn + "_iter_result.txt"
            res = {"res":set()}
            with open(textname,"r") as rf:
                for line in rf:
                    nouns = line.strip().split(",")
                    # search = Search(nouns,thresh=thresh,mode=mode,alpha=alpha,hold_multi=hold_multi,init_concepts=True)
                    search = AO(adj,nouns,thresh=thresh,mode=mode,alpha=alpha,hold_multi=hold_multi,init_concepts=True,normal_num=normal_num,beta=beta)
                    search.run()
                    # print(line.strip())
                    # print("cache size",len(cache))
                    for c in search.concepts:
                        res["res"].add(c)
                    # res[line.strip()] = {"concepts":search.concepts,"detail":search.i_prob,"score":search.score/len(search.instances)}
                    res[line.strip()] = {"concepts":search.concepts,"detail":search.i_prob,"score":search.score} 
        
            res["res"] = list(res["res"])
            json.dump(res,open(dirname+"{}_sample_pmi.json".format(tn),"w"),indent=2,ensure_ascii=False)
            json.dump(file_cache.cache,open(dirname+"{}_sample_pmi_score.json".format(tn),"w"),indent=2,ensure_ascii=False)
            print("{} has beed processed, {} results.".format(tn,str(len(res)-1)))
        end = time.time()
        print("dealing {} costs {} minutes\.n".format(adj,str((end-start)/60)))

def make_concept_npmi(thresh=40,topk=2,normal_num=1.0,beta=0.5,parse=False):
    # adjs = ["beautiful","cute","dangerous","expensive","famous","poor","popular","strong","successful","traditional"]
    # for adj in adjs:
    for d in os.listdir("./result/adjss"):
        dn = "./result/adjss/"+d
        if not os.path.isdir(dn):continue 
        adj = d
        
        cache.empty()
        sample_cache.empty()
        file_cache.empty()
        load_file_cache(os.path.join("./result/adjss/{}".format(adj),"normal_sample_pmi_score.json"))

        start = time.time()
        print("dealing {} sample:".format(adj))
        dirname = "./result/adjss/{}/".format(adj)
        test_txts = ["normal"]
        for tn in test_txts:
            if os.path.exists(dirname+"{}_sample_npmi.json".format(tn)):
                continue
            textname = dirname + tn + "_iter_result.txt"
            res = {"res":set()}
            with open(textname,"r") as rf:
                for line in rf:
                    nouns = line.strip().split(",")
                    # npmi_module = NPMI(nouns,thresh=thresh,topk=topk)
                    npmi_module = NPMIAO(adj,nouns,thresh=thresh,topk=topk,normal_num=normal_num,beta=beta,parse=parse)
                    npmi_module.run()
                    # print(line.strip())
                    # print("cache size",len(cache))
                    for c in npmi_module.concepts:
                        res["res"].add(c)
                    res[line.strip()] = {"concepts":npmi_module.concepts,"detail":npmi_module.i_prob,"score":npmi_module.score}
        
            res["res"] = list(res["res"])
            json.dump(res,open(dirname+"{}_npmi_sample.json".format(tn),"w"),indent=2,ensure_ascii=False)
            json.dump(file_cache.cache,open(dirname+"{}_sample_npmi_score.json".format(tn),"w"),indent=2,ensure_ascii=False)
            print("{} has been processed, {} results.".format(tn,str(len(res)-1)))
        end = time.time()
        print("dealing {} costs {} minutes.\n".format(adj,str((end-start)/60)))

def make_concept_bayes(thresh=40,topk=3,normal_num=1.0,beta=0.5,parse=False):
    adjs = ["beautiful","cute","dangerous","expensive","famous","poor","popular","strong","successful","traditional"]
    for adj in adjs:
    # for d in os.listdir("./result/adjss"):
    #     dn = "./result/adjss/"+d
    #     if not os.path.isdir(dn):continue 
    #     adj = d
        
        cache.empty()
        sample_cache.empty()
        file_cache.empty()
        load_file_cache(os.path.join("./result/adjs/{}".format(adj),"normal_sample_npmi_score.json"))

        start = time.time()
        print("dealing {} sample:".format(adj))
        dirname = "./result/adjs/{}/".format(adj)
        test_txts = ["normal"]
        for tn in test_txts:
            # if os.path.exists(dirname+"{}_sample_npmi.json".format(tn)):
            #     continue
            textname = dirname + tn + "_iter_result.txt"
            res = {"res":set()}
            with open(textname,"r") as rf:
                for line in rf:
                    nouns = line.strip().split(",")
                    # bayes_module = Bayes(nouns,thresh=thresh,topk=topk)
                    bayes_module = BayesAO(adj,nouns,thresh=thresh,topk=topk,normal_num=normal_num,beta=beta,parse=parse)
                    bayes_module.run()
                    print(line.strip())
                    print("cache size",len(cache))
                    for c in bayes_module.concepts:
                        res["res"].add(c)
                    res[line.strip()] = {"concepts":bayes_module.concepts,"detail":bayes_module.i_prob,"score":bayes_module.score}
        
            res["res"] = list(res["res"])
            json.dump(res,open(dirname+"{}_bayes_sample.json".format(tn),"w"),indent=2,ensure_ascii=False)
            json.dump(file_cache.cache,open(dirname+"{}_bayes_sample_score.json".format(tn),"w"),indent=2,ensure_ascii=False)
            print("{} has been processed, {} results.".format(tn,str(len(res)-1)))
        end = time.time()
        print("dealing {} costs {} minutes.\n".format(adj,str((end-start)/60)))

if __name__ == '__main__':
    make_concept()
    make_concept_npmi()
    make_concept_bayes()