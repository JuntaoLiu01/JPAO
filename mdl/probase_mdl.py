import os
import json
import math
import copy
import random

class ConceptTree:
    def __init__(self,concept_tree):
        if isinstance(concept_tree,dict):
            self.concept_tree = concept_tree
        else:
            try:
                self.concept_tree = json.load(open(concept_tree,"r"))
            except:
                print("Please use dict or file path of concept tree")

    def get_prob(self,node):
        if node not in self.concept_tree:
            return None
        return self.concept_tree[node]["prob"]

    def get_cond_prob(self,n1,n2):
        if n1 not in self.concept_tree or n2 not in self.concept_tree:
            return None
        if n2 in self.concept_tree[n1]["children"]:
            return self.concept_tree[n1]["children"][n2]
        if n2 in self.concept_tree[n1]["parents"]:
            return self.concept_tree[n1]["parents"][n2]
        return None
        
    def get_parents(self,node):
        if node not in self.concept_tree:
            return None
        return self.concept_tree[node]["parents"].keys()
    
    def get_child_num(self,node):
        if node not in self.concept_tree:
            return 0
        return len(self.concept_tree[node]["children"])

class MDL:
    def __init__(self,instances,concepts,mode="2P",alpha=0.5,**kwargs):
        self.instances = tuple(instances)
        self.concepts  = concepts
        self.mode = mode
        self.alpha = alpha
        self.maxf = float("inf")
        self.score = float("inf")

        self.c_length = None
        if "c_length" in kwargs:
            self.c_length = kwargs["c_length"]

        self.i_prob = dict()
        if self.mode == "2P":
            self.ci_prob = None
            self.choose_concept_2P()
        else:
            self.choose_concept_NML()

        self.c_prob = dict()
        self.compute_concept()
        self.score = self.compute_length(self.i_prob)

    def choose_concept_2P(self):
        for i in self.instances:
            i_prob = CT.get_prob(i)
            if i:
                li = -math.log(i_prob)
                c_index,min_i = "SELF",li
            else:
                c_index,min_i = "INIT",self.maxf
            for c in self.concepts:
                ci_prob = CT.get_cond_prob(c,i)
                if ci_prob:
                    li = math.log(len(self.concepts)) - math.log(ci_prob) * math.log2(1.0+len(c.split(" ")) * 0.8)
                    if li < min_i:
                        c_index,min_i = c,li
            self.i_prob[i] = (c_index,min_i)
    
    def update_concept_2P(self,c,mode="add",new_c=None):
        i_prob = dict()
        if mode == "add":
            for i in self.instances:
                i_prob[i] = (self.i_prob[i][0],self.i_prob[i][1])
                ci_prob = CT.get_cond_prob(c,i)
                if ci_prob:
                    # li = math.log(len(self.concepts)) - math.log(ci_prob) * math.log2(1.0+len(c.split(" ")) * 0.8)
                    li = math.log(len(self.concepts)) - math.log(ci_prob)
                    if li < self.i_prob[i][1]:
                        i_prob[i] = (c,li)  
            return i_prob
        
        elif mode == "remove":
            for i in self.instances:
                # if self.i_prob[i][0] != c:
                #    i_prob[i] = (self.i_prob[i][0],self.i_prob[i][1])
                # else:
                prob = CT.get_prob(i)
                if i:
                    li = -math.log(prob)
                    c_index,min_i = "SELF",li
                else:
                    c_index,min_i = "INIT",self.maxf
                for cc in self.concepts:
                    ci_prob = CT.get_cond_prob(cc,i)
                    if ci_prob:
                        # li = math.log(len(self.concepts)) - math.log(ci_prob) * math.log2(1.0+len(c.split(" ")) * 0.8)
                        li = math.log(len(self.concepts)) - math.log(ci_prob)
                        if li < min_i:
                            c_index,min_i = cc,li
                i_prob[i] = (c_index,min_i)
            return i_prob
        
        else:
            for i in self.instances:
                if self.i_prob[i][0] != c:
                    i_prob[i] = (self.i_prob[i][0],self.i_prob[i][1])
                    ci_prob = CT.get_cond_prob(new_c,i)
                    if ci_prob:
                        # li = math.log(len(self.concepts)) - math.log(ci_prob) * math.log2(1.0+len(c.split(" ")) * 0.8)
                        li = math.log(len(self.concepts)) - math.log(ci_prob)
                        if li < self.i_prob[i][1]:
                            i_prob[i] = (new_c,li)
                                        
                else:
                    prob = CT.get_prob(i)
                    if i:
                        li = -math.log(prob)
                        c_index,min_i = "SELF",li
                    else:
                        c_index,min_i = "INIT",self.maxf
                    for cc in self.concepts:
                        ci_prob = CT.get_cond_prob(cc,i)
                        if ci_prob:
                            # li = math.log(len(self.concepts)) - math.log(ci_prob) * math.log2(1.0+len(c.split(" ")) * 0.8)
                            li = math.log(len(self.concepts)) - math.log(ci_prob)
                            if li < min_i:
                                c_index,min_i = cc,li
                    i_prob[i] = (c_index,min_i)
            return i_prob

    def choose_concept_NML(self):
        self.ci_prob = dict()
        for i in self.instances:
            c_index,max_i = "INIT",0.0
            for c in self.concepts:
                ci_prob = CT.get_cond_prob(c,i)
                if ci_prob and ci_prob > max_i:
                        c_index,max_i = c,ci_prob
            self.ci_prob[i] = (c_index,max_i)

        all_ci_prob = sum([x[1] for x in self.ci_prob.values()])
        for i in self.instances:
            prob = CT.get_prob(i)
            if prob:
                c_index,min_i = "SELF",-math.log(prob)
            else:
                c_index,min_i = "INIT",self.maxf
            if all_ci_prob > 1e-6 and self.ci_prob[i][1] > 1e-6 and -math.log(self.ci_prob[i][1]/all_ci_prob) < min_i:
                c_index,min_i = self.ci_prob[i][0],-math.log(self.ci_prob[i][1]/all_ci_prob)
            self.i_prob[i] = (c_index,min_i)

    def update_concept_NML(self,c,mode="add",new_c=None):
        ci_prob = dict()
        i_prob = dict()
        if mode == "add":
            for i in self.instances:
                cci_prob = CT.get_cond_prob(c,i)
                if cci_prob and cci_prob > self.ci_prob[i][1]:
                    ci_prob[i] = (c,cci_prob)
                else:
                    ci_prob[i] = (self.ci_prob[i][0],self.ci_prob[i][1])
    
        elif mode == "remove":
            for i in self.instances:
                if self.ci_prob[i][0] != c:
                    ci_prob[i] =  (self.ci_prob[i][0],self.ci_prob[i][1])
                else:
                    c_index,max_i = "INIT",0.0
                    for cc in self.concepts:
                        cci_prob = CT.get_cond_prob(cc,i)
                        if cci_prob and cci_prob > max_i:
                            c_index,max_i = cc,cci_prob
                    ci_prob[i] = (c_index,max_i)
        else:
            for i in self.instances:
                if self.ci_prob[i][0] != c:
                    cci_prob = CT.get_cond_prob(new_c,i)
                    if cci_prob and cci_prob > self.ci_prob[i][1]:
                        ci_prob[i] = (new_c,cci_prob)
                    else:
                        ci_prob[i] = (self.ci_prob[i][0],self.ci_prob[i][1])
                else:
                    c_index,max_i = "INIT",0.0
                    for cc in self.concepts:
                        cci_prob = CT.get_cond_prob(cc,i)
                        if cci_prob and cci_prob > max_i:
                            c_index,max_i = cc,cci_prob
                    ci_prob[i] = (c_index,max_i)
        
        all_ci_prob = sum([x[1] for x in ci_prob.values()])
        for i in self.instances:
            prob = CT.get_prob(i)
            if prob:
                c_index,min_i = "SELF",-math.log(prob)
            else:
                c_index,min_i = "INIT",self.maxf
            if all_ci_prob > 1e-6 and ci_prob[i][1] > 1e-6 and -math.log(ci_prob[i][1]/all_ci_prob) < min_i:
                c_index,min_i = ci_prob[i][0],-math.log(ci_prob[i][1]/all_ci_prob)
            i_prob[i] = (c_index,min_i)
        return i_prob,ci_prob

    def compute_concept(self,):
        for c in self.concepts:
            if self.c_length:
                self.c_prob[c] = self.c_length[c]
            else:
                self.c_prob[c] = -math.log(CT.get_prob[c])

    def update_concept(self,c,mode="add",new_c=None):
        if mode == "add":
            if self.c_length:
                self.c_prob[c] = self.c_length[c]
            else:
                self.c_prob[c] = -math.log(CT.get_prob(c))
        elif mode == "remove":
            self.c_prob.pop(c)
        else:
            self.c_prob.pop(c)
            if self.c_length:
                self.c_prob[new_c] = self.c_length[new_c]
            else:
                self.c_prob[new_c] = -math.log(CT.get_prob(new_c))
        
    def compute_length(self,i_prob):
        mdl = self.alpha * sum([x for x in self.c_prob.values()])+\
            (1-self.alpha) * sum([x[1] for x in i_prob.values()])
        return mdl

class Search(MDL):
    def __init__(self,instances,thresh=120,mode="2P",alpha=0.5,max_iter=30,S_num=5,**kwargs):
        self.thresh = thresh
        self.iter_nums = 0
        self.max_iter = max_iter
        self.S = []
        ## for cut concepts
        self.S_num = S_num
        self.top_S = {}

        self.remove_instance = False
        if "remove_instance" in kwargs:
            self.remove_instance = kwargs["remove_instance"]
        self.init_concepts = False
        if "init_concepts" in kwargs:
            self.init_concepts = kwargs["init_concepts"]
        self.hold_multi = True
        if "hold_multi" in kwargs:
            self.hold_multi = kwargs["hold_multi"]

        instances = self.find_all_parents(instances)
        c_length = self.compute_concept_length()
        concepts = []
        super(Search,self).__init__(instances,concepts,mode,alpha,c_length=c_length)
    
    def compute_concept_length(self,):
        c_length = dict()
        tmp = []
        for c in self.S:
            prob = CT.get_prob(c)
            if prob:
                # c_length[c] = -math.log(prob)
                c_length[c] = -math.log(prob) * math.log2(1.0+len(c.split(" ")) * 0.6)
                tmp.append(c)
        self.S = tmp
        return c_length

    def find_all_parents(self,instances):
        c_set = dict()
        rem_i = []
        for i in instances:
            ip = CT.get_parents(i)
            f = False
            # max_p = 0.0
            # max_c = "INIT"
            for p in ip:
                if CT.get_child_num(p) >= self.thresh:
                    c_set[p] = CT.get_child_num(p)
                    f = True
                    # cp = CT.get_cond_prob(i,p) * CT.get_cond_prob(p,i)
                    # if cp > max_p:
                    #     max_p = cp
                    #     max_c = p
            if not f:
                rem_i.append(i)
            # if max_c != "INIT":
            #     concepts.add(max_c)
        if self.remove_instance:
            for i in rem_i:
                instances.remove(i)
        self.S = list(c_set.keys())
        if not self.hold_multi:
            self.S = [s for s in self.S if len(s.split()) == 1]
        # concepts = list(concepts)
        return instances

    def add_concept(self,):
        c_index = None
        min_score = float("inf")
        min_i_prob = None
        min_ci_prob = None
        for c in self.S:
            self.concepts.append(c)
            self.update_concept(c,mode="add")
            if self.mode == "2P":
                i_prob = self.update_concept_2P(c,mode="add")
                score = self.compute_length(i_prob)
                if score < min_score:
                    c_index,min_score,min_i_prob = c,score,i_prob
            else:
                i_prob,ci_prob = self.update_concept_NML(c,mode="add")
                score = self.compute_length(i_prob)
                if score < min_score:
                    c_index,min_score,min_i_prob,min_ci_prob = c,score,i_prob,ci_prob
            self.concepts.remove(c)
            self.c_prob.pop(c)

        return c_index,min_score,min_i_prob,min_ci_prob
    
    def remove_concept(self,):
        c_index = None
        min_score = float("inf")
        min_i_prob = None
        min_ci_prob = None
        old_concepts = copy.deepcopy(self.concepts)
        for c in old_concepts:
            self.concepts.remove(c)
            self.update_concept(c,mode="remove")
            if self.mode == "2P":
                i_prob = self.update_concept_2P(c,mode="remove")
                score = self.compute_length(i_prob)
                if score < min_score:
                    c_index,min_score,min_i_prob = c,score,i_prob
            else:
                i_prob,ci_prob = self.update_concept_NML(c,mode="remove")
                score = self.compute_length(i_prob)
                if score < min_score:
                    c_index,min_score,min_i_prob,min_ci_prob = c,score,i_prob,ci_prob
            self.concepts.append(c)
            self.c_prob[c] = self.c_length[c]
            
        return c_index,min_score,min_i_prob,min_ci_prob
    
    def replace_concept(self,):
        c_index = None
        min_score = float("inf")
        min_i_prob = None
        min_ci_prob = None
        old_concepts = copy.deepcopy(self.concepts)
        for s in self.S:
            for c in old_concepts:
                self.concepts.remove(c)
                self.concepts.append(s)
                self.update_concept(c,mode="relace",new_c=s)
                if self.mode == "2P":
                    i_prob = self.update_concept_2P(c,mode="replace",new_c=s)
                    score = self.compute_length(i_prob)
                    if score < min_score:
                        s_index,c_index,min_score,min_i_prob = s,c,score,i_prob
                else:
                    i_prob,ci_prob = self.update_concept_NML(c,mode="replace",new_c=s)
                    score = self.compute_length(i_prob)
                    if score < min_score:
                        s_index,c_index,min_score,min_i_prob,min_ci_prob = s,c,score,i_prob,ci_prob
                self.concepts.remove(s)
                self.concepts.append(c)
                self.c_prob[c] = self.c_length[c]
                self.c_prob.pop(s)
        return s_index,c_index,min_score,min_i_prob,min_ci_prob

    def cut_concepts(self):
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
        self.S =  tmp_x
        return c_index,min_score,min_i_prob,min_ci_prob

    def run(self,):
        if self.init_concepts:
            c,score,i_prob,ci_prob = self.cut_concepts()
            ## score may equal to the initial score
            if score >= self.score:
                return
            self.concepts.append(c)
            self.score = score
            self.i_prob = i_prob
            self.c_prob[c] = self.c_length[c]
            self.S.remove(c)
            self.ci_prob = ci_prob

        while True:
            if self.iter_nums > self.max_iter:
                return
            if self.S == []:return
            if self.concepts == []:
                c,score,i_prob,ci_prob = self.add_concept()
                ## score may equal to the initial score
                if score >= self.score:
                    return
                self.concepts.append(c)
                self.score = score
                self.i_prob = i_prob
                self.c_prob[c] = self.c_length[c]
                self.S.remove(c)
                self.ci_prob = ci_prob
                self.iter_nums += 1
            else:
                add_c,add_score,add_i_prob,add_ci_prob = self.add_concept()
                rem_c,rem_score,rem_i_prob,rem_ci_prob = self.remove_concept()
                rpl_s,rpl_c,rpl_score,rpl_i_prob,rpl_ci_prob = self.replace_concept()
                self.iter_nums += 1
                if add_score <= rem_score and add_score <= rpl_score:
                    if add_score >= self.score:
                        return
                    self.concepts.append(add_c)
                    self.score = add_score
                    self.i_prob = add_i_prob
                    self.S.remove(add_c)
                    self.ci_prob = add_ci_prob
                    self.c_prob[add_c] = self.c_length[add_c]
                elif rem_score <= add_score and rem_score <= rpl_score:
                    if rem_score >= self.score:
                        return
                    self.concepts.remove(rem_c)
                    self.score = rem_score
                    self.i_prob = rem_i_prob
                    self.ci_prob = rem_ci_prob
                    self.c_prob.pop(rem_c)
                else:
                    if rpl_score >= self.score:
                        return
                    self.concepts.append(rpl_s)
                    self.concepts.remove(rpl_c)
                    self.score = rpl_score
                    self.i_prob = rpl_i_prob
                    self.ci_prob = rpl_ci_prob
                    self.S.remove(rpl_s)
                    self.c_prob[rpl_s] = self.c_length[rpl_c]
                    self.c_prob.pop(rpl_c)

class NPMI:
    def __init__(self,instances,thresh=40,topk=10):
        self.instances = tuple(instances)
        self.thresh = thresh
        self.topk = topk
        self.S = dict()
        self.i_prob = dict()
        self.concepts = []
        self.find_all_parents()

    def find_all_parents(self,):
        for i in self.instances:
            self.S[i] = []
            ip = CT.get_parents(i)
            for p in ip:
                if CT.get_child_num(p) >= self.thresh:
                    self.S[i].append(p)

    def run(self,):
        concepts = set()
        score = []
        for i in self.instances:
            self.i_prob[i] = []
            for p in self.S[i]:
                p_score = self.compute_npmi(p,i)
                if p_score > 0:
                    self.i_prob[i].append([p,p_score])
            self.i_prob[i] = sorted(self.i_prob[i],key=lambda x:x[1],reverse=True)
            self.i_prob[i] = self.i_prob[i][:self.topk]
            if self.i_prob[i] != []:
                concepts.add(self.i_prob[i][0][0])
                ## get 1./npmi for converge
                score.append(1./self.i_prob[i][0][1])
        self.concepts = list(concepts)
        self.score = sum(score)/len(score)

    ## needs to computed
    @staticmethod
    def compute_npmi(c,n):
        p_n_c = CT.get_cond_prob(c,n)
        p_n = CT.get_prob(n)
        p_nc = CT.get_prob(c) * p_n_c
        return   (math.log(p_n_c)-math.log(p_n)) / (-math.log(p_nc))
    
class Bayes:
    def __init__(self,instances,thresh=40,topk=3):
        self.instances = tuple(instances)
        self.thresh = thresh
        self.topk = topk
        self.S = []
        self.i_prob = {}
        self.concepts = []
        self.score = 0.
        self.find_all_parents()

    def find_all_parents(self,):
        all_S = set()
        for index,i in enumerate(self.instances):
            c_set = set()
            ip = CT.get_parents(i)
            for p in ip:
                if CT.get_child_num(p) >= self.thresh:
                    c_set.add(p)
            all_S = (all_S|c_set) if index == 0 else (all_S & c_set)
        self.S = list(all_S)

    def run(self,):
        if self.S == []:
            return
        tmp = []
        for s in self.S:
            score = CT.get_prob(s)
            for i in self.instances:
                c_score = CT.get_cond_prob(s,i)
                score *= c_score
            tmp.append((s,score))
        tmp = sorted(tmp,key=lambda x:x[1],reverse=True)

        self.concepts = [tmp[0][0]]
        self.score = tmp[0][1]
        self.score = -math.log(self.score)/len(self.instances)
        self.i_prob = {i:tmp[:self.topk] for i in self.instances}
        
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
CT = ConceptTree(os.path.join(data_dir,"./data/p_concepts.json"))

if __name__ == '__main__':
    ## test_npmi
    # instances = ["day","week","hour","month","year"]
    # instances = ["texas","florida","california","nevada","arizona"]
    instances = ["scallops","oysters","lobsters","clams","shrimps"]

    # print(CT.concept_tree["scallops"]["parents"]["shellfish"])
    # print(CT.concept_tree["shellfish"]["children"]["scallops"])
    
    bayes_module = Bayes(instances,thresh=40,topk=3)
    bayes_module.run()
    print(bayes_module.i_prob)
    print(bayes_module.concepts)
    print(bayes_module.score)


    # npmi_module = NPMI(instances,thresh=40,topk=1)
    # npmi_module.run()
    # print(npmi_module.i_prob)
    # print(npmi_module.score)
    # print(npmi_module.concepts)

    # exp_data = json.load(open("./case_exp.json","r"))
    # exp_data = [ [[],["nets","suns","clippers"]] ]
    # exp_data = [ [[],["goldfinch","bullfinch","thrushes"]] ]

    # exp_data = [ [[],["children","women","infants"]] ]
    # for i,d in enumerate(exp_data):
    #     cur_c = d[0]
    #     cur_i = d[1]
    #     print("instances:",cur_i)
    #     print("concepts:",cur_c)
    #     search = Search(cur_i,thresh=40,alpha=0.6,mode="2P")
    #     print(search.S)
    #     search.run()
    #     print(search.concepts)
    #     print(search.i_prob)

    # res = dict()
    # data = json.load(open("cases/mul.json","r"))
    # for d in data.keys():
    #     nouns = d.strip().split(",")
    #     search = Search(nouns,thresh=40,mode="2P",alpha=0.6,hold_multi=True)
    #     search.run()
    #     res[d.strip()] = {"concepts":search.concepts,"detail":search.i_prob,"score":search.score/len(search.instances)}
    # json.dump(res,open("cases/mul_res_2.json","w"),indent=2,ensure_ascii=False)

    # for node in ["concept","term","characteristic","bird","flower"]:
    #     print(node)
    #     print("prob",CT.concept_tree[node]["prob"],"children",len(CT.concept_tree[node]["children"]),"parents",len(CT.concept_tree[node]["parents"]))
    # for node in CT.concept_tree:
    #     if len(CT.concept_tree[node]["parents"]) == 0:
    #         print(node)
    #         print("prob",CT.concept_tree[node]["prob"],"children",len(CT.concept_tree[node]["children"]),"parents",len(CT.concept_tree[node]["parents"]))

    # print(CT.get_prob("goldfinch"))
    # print(CT.get_prob("bullfinch"))
    # print(CT.get_prob("thrushes"))

    # print(CT.get_prob("common bird"))
    # print(CT.get_cond_prob("common bird","goldfinch"))
    # print(CT.get_cond_prob("common bird","bullfinch"))
    # print(CT.get_cond_prob("common bird","thrushes"))

    # print(CT.get_prob("bird"))
    # print(CT.get_cond_prob("bird","goldfinch"))
    # print(CT.get_cond_prob("bird","bullfinch"))
    # print(CT.get_cond_prob("bird","thrushes"))
    
