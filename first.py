import pandas as pd
import xgboost as xgb

data = pd.read_csv('sort_train.csv')
data = pd.DataFrame(data[:50])

x_now = []
y_now = []
x_1 = []
y_1 = []
x_2 = []
y_2 = []
x_3 = []
y_3 = []
x_4 = []
y_4 = []
x_5 = []
y_5 = []



for us_id in data.user_id:
    equal_rows = data.loc[data.user_id == us_id]
    print(equal_rows.columns)
    for rows in equal_rows.values.tolist():
        x_now.append([rows[4], rows[2]])
        y_now.append([[rows[0]]])
        for row in equal_rows.values.tolist():
            if rows[3] == row[3]-1:
                x_1.append([rows[4], rows[2]])
                y_1.append([row[0]])
            if rows[3] == row[3]-2:
                x_2.append([rows[4], rows[2]])
                y_2.append([row[0]])
            if rows[3] == row[3]-3:
                x_3.append([rows[4], rows[2]])
                y_3.append([row[0]])
            if rows[3] == row[3]-4:
                x_4.append([rows[4], rows[2]])
                y_4.append([row[0]])
            if rows[3] == row[3]-5:
                x_5.append([rows[4], rows[2]])
                y_5.append([row[0]])



xgb_params = {
    'eta': 0.5,
    'gamma': 0.3,
    'max_depth': 12,
    'min_child_weight': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'lambda': 0.07,
    'alpha': 0.5,
    'tree_method': 'approx',
    'objective': 'reg:gamma',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain_now = xgb.DMatrix(x_now, y_now)
model_now = xgb.train(xgb_params, dtrain_now)
model_now.save_model('now.model')

dtrain_1 = xgb.DMatrix(x_1, y_1)
model_1 = xgb.train(xgb_params, dtrain_1)
model_1.save_model('1.model')

dtrain_2 = xgb.DMatrix(x_2, y_2)
model_2 = xgb.train(xgb_params, dtrain_2)
model_2.save_model('2.model')

dtrain_3 = xgb.DMatrix(x_3, y_3)
model_3 = xgb.train(xgb_params, dtrain_3)
model_3.save_model('3.model')

dtrain_4 = xgb.DMatrix(x_4, y_4)
model_4 = xgb.train(xgb_params, dtrain_4)
model_4.save_model('4.model')

dtrain_5 = xgb.DMatrix(x_5, y_5)
model_5 = xgb.train(xgb_params, dtrain_5)
model_5.save_model('5.model')
