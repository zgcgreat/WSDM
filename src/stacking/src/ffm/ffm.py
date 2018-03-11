import subprocess
from datetime import datetime


path = '../../output/stack-data/ffm/'

start = datetime.now()

# 训练
cmd = './ffm-train -p {save}valid5.ffm -l 0.002 -k 4 -t 10 -s 8 -r 0.002 {save}train5.ffm ' \
      '{save}model'.format(save=path)
subprocess.call(cmd, shell=True)
# 预测
cmd = './ffm-predict {save}test.ffm {save}model {save}test.out'.format(save=path)
subprocess.call(cmd, shell=True)

with open(path + 'submission.csv', 'w') as fo:
    fo.write('msno,is_churn\n')
    for i, row in enumerate(open(path + 'test.out'), start=1):
        fo.write('{0},{1}'.format(i, row))

# cmd = 'rm {path}model {path}test.out'.format(path=path)
# subprocess.call(cmd, shell=True)

print('时间: {0}'.format(datetime.now() - start))
