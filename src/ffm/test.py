def function(arg, fo1):
    file = open('../../output/{0}/validation.csv'.format(arg), 'r')
    next(file)
    clks = 0
    ids = []
    for line in file:
        id = line.split(',')[0]
        clk = line.split(',')[1]
        if int(clk) == 1:
            clks += 1
            ids.append(id)
    file.close()
    print(clks)
    print(len(ids))

    fo = open('../../output/{0}/{0}-truth.csv'.format(arg), 'w')
    fo.write('Id,Prediction\n')
    f2 = open('../../output/{0}/submission.csv'.format(arg), 'r')
    next(f2)
    pred = 0
    true = 0
    pred1 = 0
    pred2 = 0
    pred3 = 0
    true1 = 0
    true2 = 0
    true3 = 0
    for line in f2:
        id = line.replace('\n', '').split(',')[0]
        ctr = line.replace('\n', '').split(',')[1]
        if id in ids:
            fo.write(id + ',' + str(ctr) + '\n')
        if 0 <= float(ctr) < 0.05:
            pred1 += 1
            if id in ids:
                true1 += 1
        if 0.05 <= float(ctr) < 0.2:
            pred3 += 1
            if id in ids:
                true3 += 1
        if 0.2 <= float(ctr) < 0.5:
            pred2 += 1
            if id in ids:
                true2 += 1
        if float(ctr) >= 0.5:
            pred += 1
            if id in ids:
                true += 1
    fo1.write('{0}-true'.format(arg) + ',' + str(true1) + ',' + str(true3) + ',' + str(true2) + ',' + str(true) + '\n')
    fo1.write('{0}-pred'.format(arg) + ',' + str(pred1) + ',' + str(pred3) + ',' + str(pred2) + ',' + str(pred) + '\n')
    f2.close()
    fo.close()

if __name__ == '__main__':
    fo1 = open('../../output/result.csv', 'w')
    fo1.write('method,0-0.05,0.05-0.2,0.2-0.5,0.5-1\n')
    args = ['ffm', 'fm', 'vw']
    for arg in args:
        function(arg, fo1)
    fo1.close()


# print('0-0.05 预测为1的个数:', pred1)
# print('0-0.05 真实为1的个数:', true1)
#
# print('0.05-0.2 预测为1的个数:', pred3)
# print('0.05-0.2 真实为1的个数:', true3)
#
# print('0.2-0.5 预测为1的个数:', pred2)
# print('0.2-0.5 真实为1的个数:', true2)
#
# print('>0.5 预测为1的个数:', pred)
# print('>0.5 真实为1的个数:', true)

