# _*_ coding: utf-8 _*_

import math
import sys
import subprocess

data_path = '../../output/fm/'
result_path = '../../output/fm/'

cmd = './libFM -task r -train {train} -test {test} -out {out} -method mcmc -learn_rate 0.2 -dim \'1,1,8\' -iter 150 '\
      '-validation {train}'.format(train=result_path + 'train.fm', test=result_path + 'test.fm', out=result_path + 'preds.txt')
subprocess.call(cmd, shell=True, stdout=sys.stdout)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

msno = []
fi = open('../../data/test.csv', 'r')
next(fi)
for line in fi:
    msno.append(line.split(',')[0])
fi.close()


with open(result_path + 'submission.csv', 'w') as outfile:
    outfile.write('msno,is_churn\n')
    for t, line in enumerate(open(result_path + 'preds.txt')):
        # outfile.write('{0},{1}\n'.format(t, sigmoid(float(line.rstrip()))))
        outfile.write('{0},{1}\n'.format(msno[t], float(line.rstrip())))

# cmd = 'rm {0}preds.txt'.format(result_path)
# subprocess.call(cmd, shell=True)


