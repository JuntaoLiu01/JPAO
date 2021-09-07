import os
import json
import gzip
import requests
import random
from nltk.stem import WordNetLemmatizer

def save_item(wf,item_dict):
    res = sorted(item_dict.items())
    for x in item_dict.items():
        try:
            wf.write("\t".join(x[0])+"\t"+x[1])
            wf.write("\n")
        except:
            pass

def parse_file():
    """
    Function:
        extract adjective-noun pair from Google N-Grams corpus.
        Corpus download link: http://commondatastorage.googleapis.com/books/syntactic-ngrams/index.html
        craw_data.py is our python script for downloading corpus.
    Output:
        serveral txt files saving (adjective,noun,cooccurence times) pairs.
    """
    dst_dir = "./data/adj_noun"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    src_dir = ["Biarcs"]
    for title in src_dir:
        print("parsing {}".format(title))
        cur_dir = os.path.join("./data/English_All",title)
        for filename in os.listdir(cur_dir):
            if not filename.endswith("gz"):continue
            print(filename)
            file = os.path.join(cur_dir,filename)
            res = dict()
            with gzip.open(file,"r") as rf:
                for line in rf:
                    line = line.decode("utf-8").strip()
                    count = line.split("\t")[2]
                    item = line.split("\t")[1]
                    if "/JJ/amod/" in item:
                        adjs = [];nouns = []
                        for i,element in enumerate(item.split(" ")):
                            if len(element.split("/")) != 4:
                                break
                            if "/JJ/amod/" in element:
                                adjs.append((element.split("/")[0],int(element.split("/")[3]),i))
                            if "/NN" in element:
                                nouns.append((element.split("/")[0],int(element.split("/")[3]),i))
                        if adjs != [] and nouns != []:
                            for adj in adjs:
                                for noun in nouns:
                                    if adj[1] > noun[1] and adj[2] + 1 == noun[2]:
                                        res[(adj[0],noun[0])] = max(res.get((adj[0],noun[0]),"0"),count)
            
            with open(os.path.join(dst_dir,title+".txt"),"a") as wf:
                save_item(wf,res)

def remove_num():
    """
    Function:
        filter out numbers in (adjective,noun,cooccurence) pairs.
    Output:
        several txt file saving (adjective,noun,cooccurence).
    """
    dst_dir = "./data/adj_letter"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    src_dir = "./data/adj_noun"
    for filename in os.listdir(src_dir):
        if not filename.endswith(".txt"):continue
        print(filename)
        res = dict()
        file = os.path.join(src_dir,filename)
        with open(file,"r") as rf:
            for line in rf:
                adj,noun,count = line.strip().split("\t")
                if not adj.isalpha() or not noun.isalpha():
                    continue
                res[(adj,noun)] = str(int(res.get((adj,noun),"0")) + int(count))
    
        with open(os.path.join(dst_dir,filename),"w") as wf:
            save_item(wf,res)

def stem_noun():
    dst_dir = "./data/adj_stem"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    src_dir = "./data//adj_letter"
    lemmatizer = WordNetLemmatizer()
    wf = open("./data/stem_nouns.txt")
    for filename in os.listdir(src_dir):
        if not filename.endswith(".txt"):continue
        print(filename)
        file = os.path.join(src_dir,filename)
        res = dict()
        with open(file,"r") as rf:
            for line in rf:
                adj,noun,count = line.strip().split("\t")
                raw_noun = lemmatizer.lemmatize(noun)
                if (adj,raw_noun) not in res:
                    res[(adj,raw_noun)] = count
                else:
                    wf.write(noun+"\t"+raw_noun+"\n")
        with open(os.path.join(dst_dir,filename),"w") as cwf:
            save_item(cwf,res)

def count_word_freq(src_dir="./data/adj_letter"):
    adj_dict = dict()
    noun_dict = dict()
    for filename in os.listdir(src_dir):
        if not filename.endswith(".txt"):continue
        print(filename)
        file = os.path.join(src_dir,filename)
        with open(file,"r") as rf:
            for line in rf:
                adj,noun,_ = line.strip().split("\t")
                adj_dict[adj] = adj_dict.get(adj,0)+1
                noun_dict[noun] = noun_dict.get(noun,0)+1
    json.dump(adj_dict,open("./data/adj_freq.json","w"))
    json.dump(noun_dict,open("./data/noun_freq.json","w"))

def translate(word):
    base_url = "http://dict-co.iciba.com/api/dictionary.php"
    res_type = "json"
    key = "54A9DE969E911BC5294B70DA8ED5C9C4"
    res = requests.get(base_url,params={"type":res_type,"w":word,"key":key})
    res_str = res.content.decode(res.encoding)
    res_dict = json.loads(res_str)
    if "symbols" in res_dict:
        pos_set = set()
        for symbol in res_dict["symbols"]:
            if "parts" in symbol:
                for part in symbol["parts"]:
                    pos_set.add(part["part"])
        pos_list = list(pos_set)
        if len(pos_list) == 0:
            return None 
        return pos_list
    return None

def build_word_pos():
    """
    Function:
        collect all Pos-of-Tag information for each adjective and noun.
    Output:
        word_pos.json saving word Pos-of-Tag information.
    """
    if os.path.exists("./data/word_pos.json"):
        word_pos = json.load(open("./data/word_pos.json","r"))
        None_set = json.load(open("./data/no_pos.json","r"))
    else:
        word_pos = dict()
        None_set = dict()
    print(len(word_pos))
    src_data = ["./data/adj_freq.json"]
    # "./data/noun_freq.json"
    try:
        for file in src_data:
            print(file)
            word_dict = json.load(open(file,"r"))
            for i,word in enumerate(word_dict):
                if i % 1000 == 0 and i != 0:
                    print(i)
                if word not in None_set and word not in word_pos :
                    pos = translate(word)
                    if pos: 
                        word_pos[word] = pos
                    else:
                        None_set.append(word)
        print(len(word_pos))
        json.dump(word_pos,open("./data/word_pos.json","w"),ensure_ascii=False)
        json.dump(None_set,open("./data/no_pos.json","w"))
    except Exception as e:
        print(e)
        print(len(word_pos))
        json.dump(word_pos,open("./data/word_pos.json","w"),ensure_ascii=False)
        json.dump(None_set,open("./data/no_pos.json","w"))
     
def filter_by_pos():
    """
    Function:
        filtering adjective and noun by Pos-of-Tag information.
    Output:
        several txt files saving (adjective,noun) pairs.
    """
    word_pos = json.load(open("./data/word_pos.json","r"))
    dst_dir = "./data/adj_pos"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    src_dir = "./data/adj_letter"
    for filename in os.listdir(src_dir):
        if not filename.endswith(".txt"):continue
        print(filename)
        file = os.path.join(src_dir,filename)
        wf = open(os.path.join(dst_dir,filename),"w")
        with open(file,"r") as rf:
            for line in rf:
                adj,noun,count = line.strip().split("\t")
                if adj not in word_pos or noun not in word_pos:
                    continue
                if "adj." not in word_pos[adj]:
                    continue 
                if "n." not in word_pos[noun]:
                    continue
                wf.write(line.strip()+"\n")
 
def merge_all():
    """
    Function:
        merging all txt files.
    Output:
        all.txt saving (adjective,noun) pairs.
    """
    src_dir = "./data/adj_pos"
    res = dict()
    for filename in os.listdir(src_dir):
        if not filename.endswith(".txt"):continue
        print(filename)
        file = os.path.join(src_dir,filename)
        with open(file,"r") as rf:
            for line in rf:
                adj,noun,count = line.strip().split("\t")
                res[(adj,noun)] = str(int(res.get((adj,noun),"0")) + int(count))
    print(len(res))
    with open("./data/all.txt","w") as wf:
        save_item(wf,res)

## filter noun not in concepts       
def prepare_for_mdl():
    """
    Function:
        Transforming all.txt to mdl_data.json
    Output:
        mdl_data.json saving (adjective,noun) pairs.
    """
    if not os.path.exists("./data/all.txt"):
        merge_all()
    instances = json.load(open("./mdl/data/all_instance.json","r"))
    instances = set(instances)
    res = dict()
    noun_set = set()
    try:
        with open("./data/all.txt","r") as rf:
            for i,line in enumerate(rf):
                if i % 1000 == 0 and i != 0:
                    print(i)
                adj,noun,count = line.strip().split("\t")
                if noun in noun_set:continue
                if noun not in instances:
                    noun_set.add(noun)
                    continue
                if adj not in res:
                    res[adj] = {noun:count}
                else:
                    res[adj][noun] = count
    except Exception as e:
        print(e)
    finally:
        print(len(res))
        count = sum([len(res[x]) for x in res])
        print(count/len(res))
        json.dump(res,open("./data/mdl_data.json","w"))
        noun_set = list(noun_set)
        print(len(noun_set))
        with open("./data/no_noun_1.txt","w") as wf:
            for x in noun_set:
                wf.write(x+"\n")

def filter_by_freq(filename,thresh=30):
    res = dict()
    test_data = json.load(open(filename,"r"))
    for case in test_data:
        res[case] = dict()
        for word in test_data[case]:
            if int(test_data[case][word]) < thresh:
                continue
            res[case][word] = test_data[case][word]
        if len(res[case]) == 0:
            res.pop(case)
            print(case)
        # print(len(res[case])/len(test_data[case]))
    json.dump(res,open("./data/test_data_{}.json".format(str(thresh)),"w"))

def filter_by_model(model_path,thresh=0.5):
    """
    Function:
        filtering unreasonable (adjective,noun) pair through classifier.
    Output:
        mdl_prob_data.json
    """
    import classifier.run_classify_prob as rb
    if not os.path.exists("./classifier/data/mdl_prob_data.csv"):
        import create_data as cd
        mdl_prob_data = cd.create_mdl_prob_data()
    else:
        mdl_prob_data = rb.load_data("run_all","./classifier/data/mdl_prob_data.csv")
    print("data have been prepared! start predicting!")
    pred = rb.predict(mdl_prob_data,model_path,wf=False)
    print("predicting have been done! start filtering!")
    pred_index = 0
    res = {}
    mdl_data = json.load(open("./data/mdl_data.json","r"))
    all_count = 0
    filter_count = 0
    for adj in mdl_data:
        res[adj] = {}
        for noun in mdl_data[adj]:
            if pred[pred_index][0] > thresh:
                res[adj][noun] = mdl_data[adj][noun]
                filter_count += 1
            all_count += 1
            pred_index += 1

        if len(res[adj]) == 0:
            res.pop(adj)
            # print(adj)
    print(all_count)
    print(filter_count)
    json.dump(res,open("./data/mdl_prob_data.json","w"))

if __name__ == '__main__':
    parse_file()
    remove_num()
    # count_word_freq()
    build_word_pos()
    filter_by_pos()
    merge_all()
    prepare_for_mdl()

    # filter_by_freq("./data/mdl_data.json",thresh=30)
    # filter_by_model(model_path="",thresh_hold=0.5)


