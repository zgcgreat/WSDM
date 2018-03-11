import pandas as pd
import numpy as np



from csv import DictReader
import pandas as pd
from multiprocessing import Pool, cpu_count
import numpy as np
from datetime import datetime


def parallel(df, func):
    if len(df) > 0:
        p = Pool(cpu_count())
        df = p.map(func, np.array_split(df, cpu_count()))
        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
        p.close(); p.join()
        return df


def data_clean(df):
    p = 0
    n = 0
    indices = []
    for index, row in df.iterrows():
        if abs(float(row['is_churn']) - float(row['pred'])) > 0.8:
            indices.append(index)
            if int(row['is_churn']) == 1:
                p += 1
            else:
                n += 1
    # tmp = df[indices]
    # tmp['is_churn'] = 0
    df = df.drop(indices, axis=0)
    df = df.drop(['pred'], axis=1)
    # df = pd.concat([df, tmp], axis=0)
    print(p, n, p+n)
    return df

if __name__ == '__main__':
    start = datetime.now()
    df_train = pd.read_csv('../../data/train_stacking.csv')
    df_train = df_train.drop(['msno', 'is_churn'], axis=1)
    avg = 0
    for col in df_train.columns:
        avg += df_train[col]
    df_train['pred'] = avg / len(df_train.columns)
    df_train = df_train['pred']

    train = pd.read_csv('../../data/train.csv')

    df = pd.concat([train, df_train], axis=1)

    # df = pd.read_csv(fi)
    l = len(df)
    # df = parallel(df, data_clean)
    df = data_clean(df)
    l1 = len(df)
    print(l, l1, l-l1, (l-l1)/l)
    df.to_csv('../../data/train_clean.csv', index=False)
    print(datetime.now() - start)