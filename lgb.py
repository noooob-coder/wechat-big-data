import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
def lgb_model(data,action):
    params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # GBDT算法为基础
    'objective': 'binary',  # 因为要完成预测用户是否买单行为，所以是binary，不买是0，购买是1
    'metric': 'auc',  # 评判指标
    'max_bin': 255,  # 大会有更准的效果,更慢的速度
    'learning_rate': 0.1,  # 学习率
    'num_leaves': 64,  # 大会更准,但可能过拟合
    'max_depth': -1,  # 小数据集下限制最大深度可防止过拟合,小于0表示无限制
    'feature_fraction': 0.8,  # 防止过拟合
    'bagging_freq': 5,  # 防止过拟合
    'bagging_fraction': 0.8,  # 防止过拟合
    'min_data_in_leaf': 21,  # 防止过拟合
    'min_sum_hessian_in_leaf': 3.0,  # 防止过拟合
    'header': True  # 数据集是否带表头
        }
    dataset_future=data.test_df
    dataset=data.total_df[data.column_list+[action]]
    dataset = dataset.sample(frac=1.0 / data.ACTION_SAMPLE_RATE[action], random_state=42, replace=False)  # 下采样
    d_x = dataset.iloc[:, :-1].values
    d_y = dataset[action].values
    d_future_x = dataset_future.iloc[:, :].values
    train_X, valid_X, train_Y, valid_Y = train_test_split(
        d_x, d_y, test_size=0.2, random_state=2)  # 将训练集分为训练集+验证集
    lgb_train = lgb.Dataset(train_X, label=train_Y)
    lgb_eval = lgb.Dataset(valid_X, label=valid_Y, reference=lgb_train)
    print("Training...")
    bst = lgb.train(
        params,
        lgb_train,
        categorical_feature=list(range(1, 17)),  # 指明哪些特征的分类特征
        valid_sets=[lgb_eval],
        num_boost_round=500,
        early_stopping_rounds=200)
    print("Saving Model...")
    # bst.save_model(model_file)  # 保存模型
    print("Predicting...")
    predict_result = bst.predict(d_future_x)  # 预测的结果在0-1之间，值越大代表预测用户购买的可能性越大
    return predict_result
def lgb_main(data):
    submit = pd.read_csv('./data/test_a.csv')[['userid', 'feedid']]
    for action in data.action:
        rst = lgb_model(data,action)
        submit[action] = rst
    print('正在保存结果')
    submit.to_csv("./submit_base_lgb.csv", index=False)




