import pandas as pd
from sklearn.utils import shuffle
class Dataload(object):
    def __init__(self):
        self.ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10,
                      "favorite": 10}
        self.feed_list=['authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
        self.action=["read_comment", "like", "click_avatar", "forward"]
        self.feed_info_path='./data/feed_info.csv'
        self.user_info_path='./data/user_action.csv'
        self.feed_info=pd.read_csv(self.feed_info_path)[['feedid']+self.feed_list]  #读取feedinfo一整张表
        self.user_info=pd.read_csv(self.user_info_path)[['userid','feedid']+self.action]  #读取userinfo一整张表
        self.user_ctr=self.click_rate('userid')   #user每个点击率表
        self.feed_ctr=self.click_rate('feedid')   #feed每个点击率表
        self.weight=pd.read_csv('./data/weight.csv')     #feed相关权重表
        # self.feed_emd=pd.read_csv('./data/feed_embeddings.csv')  #获取feedembedding
        self.online=pd.read_csv('./data/test_a.csv') #网上提交的内容
        self.total_df=self.df_load(self.user_info)   #将所有内容整合到一张表上
        self.test_df=self.df_load(self.online)
        self.column_list=[column for column in self.total_df if column not in self.action]  #获取除action以外的所有列的名称
    def click_rate(self,column):
        if column == 'userid':
            click_df=self.user_info[column].drop_duplicates()
        else:
            click_df=self.feed_info[column].drop_duplicates()
        for action in self.action:
            action_load=pd.read_csv(f'./data/bayes_{column}ctr_data_for_{action}.csv')[[column,'bayes']]
            action_load.columns=[column,column+'_'+action+'_bayes']
            click_df=pd.merge(click_df,action_load,on=column,how='left')
        return click_df
    def df_load(self,base_df):
        train=base_df
        train=pd.merge(train,self.feed_info,on='feedid',how='left')
        train=pd.merge(train,self.feed_ctr,on='feedid',how='left')
        train=pd.merge(train,self.weight,on='feedid',how='left')
        train=pd.merge(train,self.user_ctr,on='userid',how='left')
        train=shuffle(train)  #随机打乱
        return train
