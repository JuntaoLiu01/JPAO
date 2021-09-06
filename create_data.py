import json
import math

PAN_MIN = 0.0
PAN_MAX = 18.870263336563777
PNA_MIN = 0.0
PNA_MAX = 17.898878918140582
REP_MIN = 0.0
REP_MAX = 33.31414514660746

def create_noun_adj_dict():
    res = {}
    data = json.load(open("./data/mdl_data.json","r"))
    for adj in data:
        for noun in data[adj]:
            if not noun in res:
                res[noun] = {adj:data[adj][noun]}
            else:
                res[noun][adj] = data[adj][noun]
    json.dump(res,open("./data/mdl_data_inv.json","w"))

def create_mdl_prob_data():
    mdl_data = json.load(open("./data/mdl_data.json","r"))
    mdl_data_inv = json.load(open("./data/mdl_data_inv.json","r"))
    wf = open("./classifier/data/mdl_prob_data.csv","w")
    wf.write("adjective,label,noun,pan,pna,rep\n")
    res_data = []
    for i,adj in enumerate(mdl_data):
        if i % 1000 == 0:print(i)
        all_a = sum([int(x) for x in mdl_data[adj].values()])
        nouns = mdl_data[adj].keys()
        for n in nouns:
            all_n = sum([int(x) for x in mdl_data_inv[n].values()])
            co = int(mdl_data[adj][n])
            pan = -math.log(co/all_a)
            pna = -math.log(co/all_n)
            rep = -math.log((co * co)/(all_a * all_n))

            pan = str((pan - PAN_MIN)/(PAN_MAX-PAN_MIN))
            pna = str((pna - PNA_MIN)/(PNA_MAX-PNA_MIN))
            rep = str((rep - REP_MIN)/(REP_MAX-REP_MIN))
            wf.write(",".join([adj,n,"-1",pan,pna,rep]) + "\n")
            res_data.append([adj+" "+n,"-1",pan,pna,rep])
    return res_data

if __name__ == '__main__':
    create_noun_adj_dict()