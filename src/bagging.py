import os
import pandas as pd
from multiprocessing import Pool, cpu_count
import numpy as np
from sklearn.utils import shuffle
from _datetime import datetime

root_path = '../output/'


def parallel(df, func):
    if len(df) > 0:
        p = Pool(cpu_count())
        df = p.map(func, np.array_split(df, cpu_count()))
        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
        p.close(); p.join()
        return df


def df_merge(arg):
    path = root_path + '{}/'.format(arg)
    lgb_preds = os.listdir(path)
    lgb_preds = shuffle(np.array(lgb_preds))
    sub = pd.DataFrame()
    for t, f in enumerate(lgb_preds, start=1):
        df = pd.read_csv(path + f)
        sub = sub.append(df)
        df = []
        print(t, f)
        if t == 30:
            break
    return sub


def bagging(sub):
    sub = sub.groupby(['msno'], as_index=False).mean()
    return sub


if __name__ == '__main__':
    sub = pd.DataFrame()
    args = ['nn', 'bagging', 'lgb']
    for arg in args:
        df = df_merge(arg)
        sub = sub.append(df)
        df = []
    print(sub.head())
    print("bagging...")
    # sub = parallel(sub, bagging)
    sub = bagging(sub)
    sub.to_csv('../output/bagging/sub_bagging{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), index=False)

    print(sub.head())
    print(len(sub))