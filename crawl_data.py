import os
import json
import wget
import requests
import subprocess
from lxml import etree


class Spider:
    def get_source(self,url):
        proxies = {
                    "https": "https://127.0.0.1:1087",
                    "http": "http://127.0.0.1:1087"
                }
        headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36"}
        html = requests.get(url,headers=headers,proxies=proxies,timeout=5)
        return html.text

    def parse_info(self,response):
        data = {}
        selector = etree.HTML(response)
        h2 = selector.xpath("//*[@id='container']/h2/text()")
        all_p = selector.xpath("//*[@id='container']/p")
        all_title = []
        all_link = []
        for i in range(3,len(all_p)+1,14):
            cur_title = []
            cur_link = []
            for j in range(i,i+14):
                sub_title = selector.xpath("//*[@id='container']/p[{}]/b/text()".format(str(j)))
                sub_link = selector.xpath("//*[@id='container']/p[{}]/a/@href".format(str(j)))
                cur_title.extend(sub_title)
                cur_link.append(sub_link)
            title = h2[int((i-3)/14)]
            data[title] = {}
            for k,st in enumerate(cur_title):
                data[title][st] = cur_link[k]
        with open("./data/download_link.json","w") as wf:
            json.dump(data,wf,indent=4)
        
        # print(p)

def download(title="English All"):
    with open("./data/download_link.json","r") as rf:
        data = json.load(rf)
    download_link = data[title]
    base_dir = "./data/{}".format(title.replace(" ","_"))
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for title in download_link:
        print(title)
        cur_dir = base_dir+"/"+title.replace(" ","_")
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        for link in download_link[title]:
            print(link)
            wget.download(link,cur_dir+"/"+link.split("/")[-1],)
            src_file = cur_dir+"/"+link.split("/")[-1]
            scp_file_to_remote(src_file)
            os.remove(src_file)

def scp_file_to_remote(localsource,user="",ip="",password="",remotedest="",port=22):
    SCP_CMD_BASE = r"""
        expect -c "
        set timeout 300 ;
        spawn scp -P {port} -r {localsource} {username}@{host}:{remotedest} ;
        expect *assword* {{{{ send {password}\r }}}} ;
        expect *\r ;
        expect \r ;
        expect eof
        "
    """.format(username=user,password=password,host=ip,localsource=localsource,remotedest=remotedest,port=port)
    SCP_CMD = SCP_CMD_BASE.format(localsource = localsource)
    print("execute SCP_CMD: ",SCP_CMD)
    p = subprocess.Popen( SCP_CMD , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    p.communicate()
    os.system(SCP_CMD)   
            

if __name__ == '__main__':
    url = "http://commondatastorage.googleapis.com/books/syntactic-ngrams/index.html"
    spider = Spider()
    html = spider.get_source(url)
    spider.parse_info(html)
    download()
