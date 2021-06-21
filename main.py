from data_load import Dataload
from featureSelect import nullImportant
from deep_fm import fm_main
from lgb import lgb_model
import datetime
from multy_learn import esmm_train
from multy_learn import mmoe_train
import pandas as pd
def importance_by_lgb(data):
    path='importance_log.txt'
    f=open(path,'a+',encoding='utf-8')
    for action in data.action:
        time = datetime.datetime.now()
        f.write(str(time)+'\n')
        action_info='Action:' + action
        f.write(action_info+'\n')
        print(action_info)
        impotance=nullImportant(data.total_df,data.column_list,action)
        for item in impotance:
            f.write(str(item)+'\n')
        print(impotance)
    f.close()
def deep_fm(data):
    dense_features=['videoplayseconds']
    sparse_features=[i for i in data.column_list if i not in dense_features]
    df=fm_main(data,dense_features,sparse_features)
    df.to_csv("./submit_base_deepfm.csv", index=False)
def lgb(data):
    submit = data.test_df[['userid', 'feedid']]
    for action in data.action:
        rst = lgb_model(data,action)
        submit[action] = rst
    print('正在保存结果')
    submit.to_csv("./submit_base_lgb.csv", index=False)
def esmm(data):
    submit=esmm_train(data,3)
    submit.to_csv("./submit_base_esmm.csv", index=False)
def mmoe(data):
    submit = mmoe_train(data, 3)
    submit.to_csv("./submit_base_mmoe.csv", index=False)
if __name__ == '__main__':
    print("data loading")
    data=Dataload()
    print("loading end")
    print("importance predictind")
    importance_by_lgb(data)
    print("end predict")
    print("deep_fm_training")
    deep_fm(data)
    print("end")
    print("lgb")
    lgb(data)
    print('end')