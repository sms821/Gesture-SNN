import numpy as np

outdir = '/i3c/hpcl/sms821/Research/SpikSim/slayer_scratch/slayerPytorch/example/12_datasets/128x128'
testfile = '/i3c/hpcl/sms821/Research/SpikSim/slayer_scratch/slayerPytorch/example/12_datasets/128x128/test.txt'
outfile = '/i3c/hpcl/sms821/Research/SpikSim/slayer_scratch/slayerPytorch/example/12_datasets/128x128/test_new.txt'
test_train_ratio = 0.24
trainset = {
    0: 964, 1: 1864, 10: 4776, 2: 1883, 3: 7673,
    4: 3845, 5: 3892, 6: 2413, 7: 2443, 8: 2484,
    9: 1467
    }
testset = {
    0: 450, 1: 622, 10: 1136, 2: 558, 3: 2994,
    4: 1700, 5: 2128, 6: 1651, 7: 738, 8: 968,
    9: 934
    }

samples = np.loadtxt(testfile, dtype=str)
ofh = open(outfile, 'w')
ofh.write('#sample   #class\n')

running_count = {}
for i in range(11):
    running_count[i] = 0

for i in range(len(samples)):
    filenm, label = samples[i][0], int(samples[i][1])
    if running_count[label] > test_train_ratio * trainset[label]:
        continue

    spaceStr = ' '*(12 - 2 - len(filenm))
    writeStr = filenm + spaceStr + str(label) + '\n'
    ofh.write(writeStr)
    running_count[label] += 1
    #if i > 10:
    #    break
ofh.close()

for k, v in running_count.items():
    print(k, v)
