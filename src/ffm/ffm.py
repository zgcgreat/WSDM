import subprocess

NR_THREAD = 8

path = '../../output/ffm/'


# 训练
cmd = './ffm-train -l 0.002 -k 8 -t 3 -r 0.002 -s {nr_thread} -v 5 {save}tr.ffm ' \
      '{save}model'.format(nr_thread=NR_THREAD, save=path)
subprocess.call(cmd, shell=True)
# 预测
cmd = './ffm-predict {save}test.ffm {save}model {save}test.out'.format(save=path)
subprocess.call(cmd, shell=True)


msno = []
fi = open('../../data/test.csv', 'r')
next(fi)
for line in fi:
    msno.append(line.split(',')[0])
fi.close()


with open(path + 'submission.csv', 'w') as f:
    f.write('msno,is_churn\n')
    for i, row in enumerate(open(path + 'test.out')):
        f.write('{0},{1}'.format(msno[i], row))


# 删除中间文件
# cmd = 'rm {path}model {path}test.out'.format(path=path)
# subprocess.call(cmd, shell=True)
