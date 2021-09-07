#_*_ coding: utf-8 _*_
import os
import random
import json

def is_instance(node):
    return len(node["children"]) == 0

def is_root_concept(node):
    return len(node["parents"]) == 0

def filter_root_concept(concept_tree):
    concepts = []
    for node in concept_tree:
        if is_root_concept(concept_tree[node]):
            concepts.append(node)
    json.dump(concepts,open("./data/all_root_concept.json","w"),indent=4)
    return concepts

def filter_instance(concept_tree):
    instances = []
    for node in concept_tree:
        if is_instance(concept_tree[node]):
            instances.append(node)
    json.dump(instances,open("./data/all_instance.json","w"),indent=4)
    return instances

def compute_prob(concept_tree):
    all_count = 0
    ## 关于p(e|c)和p(c|e)的计算
    for node in concept_tree:
        count = sum([c for c in concept_tree[node]["children"].values()])
        count_p = sum([c  for c in concept_tree[node]["parents"].values()])
        all_count += count
        concept_tree[node]["count"] = count + count_p
        for child,value in concept_tree[node]["children"].items():
            concept_tree[node]["children"][child] = [value,value/count]
        for parent,value in concept_tree[node]["parents"].items():
            concept_tree[node]["parents"][parent] = [value,value/count_p]
    for node in concept_tree:
        concept_tree[node]["prob"] = concept_tree[node]["count"]/all_count
    json.dump(concept_tree,open("./data/concepts_tree.json","w"),indent=4)

def parse_file(filename):
    concept_tree = dict()
    with open(filename,"r") as rf:
        for line in rf:
            concept,instance,count = line.strip().split("\t")
            if concept not in concept_tree:
                concept_tree[concept] = {
                    "parents":{},
                    "children":{instance:int(count)}
                }
            else:
                concept_tree[concept]["children"][instance] = int(count)
            if instance not in concept_tree:
                concept_tree[instance] = {
                    "parents":{concept:int(count)},
                    "children":{}
                }
            else:
                concept_tree[instance]["parents"][concept] = int(count)
    filter_instance(concept_tree)
    filter_root_concept(concept_tree)
    compute_prob(concept_tree)

    print(concept_tree["microsoft"])
    # print(concept_tree["apple"])

def reduce_tree():
    base_tree = json.load(open("./data/concepts_tree.json","r"))
    new_tree = dict()
    # node2id = dict()
    # for i,node in enumerate(base_tree):
    #     node2id[node] = i
    for node in base_tree:
        new_tree[node] = dict()
        new_tree[node]["prob"] = base_tree[node]["prob"]
        new_tree[node]["children"] = dict()
        new_tree[node]["parents"] = dict()
        for child in base_tree[node]["children"]:
            new_tree[node]["children"][child] = base_tree[node]["children"][child][1]
        for parent in base_tree[node]["parents"]:
            new_tree[node]["parents"][parent] = base_tree[node]["parents"][parent][1]
        base_tree[node] = ""
    # id2node = dict()
    # for node,i in node2id.items():
    #     id2node[i] = node
    # json.dump(node2id,open("./data/node2id.json","w"))
    # json.dump(id2node,open("./data/id2node.json","w"))
    json.dump(new_tree,open("./data/p_concepts.json","w"))

        
if __name__ == '__main__':
    parse_file("./data/data.txt")
    reduce_tree()

