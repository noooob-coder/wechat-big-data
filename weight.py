import pandas as pd
class Weight(object):
    def __init__(self):
        self.path="./data/feed_info.csv"
        self.feed_info = pd.read_csv(self.path)[['feedid', 'manual_tag_list']]
        self.tag=self.tag_score()
    def tag_score(self):
        rst= {}
        dic={}
        self.tag_list=self.feed_info['manual_tag_list'].fillna(0)
        sum=0
        for i in self.tag_list:
            if i == 0:
                continue
            i=i.split(';')
            for line in i :
                if line not in dic.keys():
                    dic.setdefault(line, []).append(1)
                else:
                    dic[line].append(1)
            sum=sum+len(i)
        for key in dic.keys():
            num=len(dic[key])/sum
            rst.setdefault(key,num)
        return rst
    def feed_score(self):
        score_list=[]
        for tag in self.tag_list:
            if tag == 0:
                score_list.append(0)
            else:
                tag=tag.split(';')
                sum=0
                for i in tag:
                    sum=sum+self.tag[i]
                score_list.append(sum)
        return score_list
    def rst(self):
        score=self.feed_score()
        rst=self.feed_info[['feedid']]
        rst['score']=score
        rst.to_csv('weight.csv',index=False)
        return rst
w=Weight()
w.rst()
