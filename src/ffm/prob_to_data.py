
import pandas as pd

data = pd.read_csv('../../data/test.csv')
sub = pd.read_csv('../../output/ffm/submission.csv')
print(len(data), len(sub))
data = pd.concat([sub['Predicted'], data], axis=1)

data.to_csv('../../output/ffm/data.csv', index=False)

print(data.head())
