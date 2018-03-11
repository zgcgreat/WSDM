# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../data_ori"]).decode("utf8"))
test = pd.read_csv('../data_ori/sample_submission_zero.csv')
test.loc[:617459, 'is_churn'] = 1
test.to_csv('zero.csv', index=False)
# Any results you write to the current directory are saved as output.
print(test.head())

df_test = pd.read_csv('../data/test.csv')

df_test['is_churn'] = test['is_churn']
print(df_test.head())
df_test.to_csv('../data/test.csv', index=False)