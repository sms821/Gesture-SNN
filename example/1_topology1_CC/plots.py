import os
import matplotlib.pyplot as plt
import numpy as np

indir = 'config_10k_2c_64_deep'
def plot(indir):
    all_files = os.listdir(indir)

    # collect spike file names
    spike_files = []
    ann_files = []
    for f in sorted( all_files ):
        if f.endswith('.npz'):
            if 'spike' in f:
                spike_files.append(f)
            else:
                ann_activity(np.load( os.path.join(indir, f) ))

    # collect layer-wise keys
    data = np.load( os.path.join(indir, spike_files[0]) )
    layer_names = list(data.keys())

    # iterate over all layers
    for L, ln in enumerate( layer_names):
        print('\nLayer ', ln)
        data = []
        for sf in spike_files:
            if L == 0: print('Reading file ', sf)
            data.append(np.load(os.path.join(indir, sf))[ln])
        #plot_spike_raster(data, os.path.join(indir, 'raster_'+f[12:-4]+ln), ln)
        plot_spike_count(data, indir, ln)
        #if L == 1:
        #    break

def ann_activity(data_dict):
    print('Showing non-zero ratio in ANN..')
    for k, v in data_dict.items():
        nonzeros = np.nonzero(v)[0].shape
        tensor_sz = np.prod(v.shape)
        nz_ratio = nonzeros / tensor_sz
        print(k, nz_ratio)

def plot_spike_count(spike_arr, outdir, layer_name=""):
    print('Plotting spike counts')
    for i in range(len(spike_arr)):
        print('total spikes ', np.sum(spike_arr[i]))
        if len(spike_arr[i].shape) == 1:
            spike_arr[i] = np.expand_dims(spike_arr[i], axis=(0,1,2))
        elif len(spike_arr[i].shape) == 2:
            spike_arr[i] = np.expand_dims(spike_arr[i], axis=(0,1))
        elif len(spike_arr[i].shape) == 3:
            spike_arr[i] = np.expand_dims(spike_arr[i], axis=(0))
        elif len(spike_arr[i].shape) == 4:
            pass
        else:
            print('Invalid data shape ', data.shape)
            return
    titles = ['aedat aggregated', 'poisson', 'aedat']
    max_counts = []
    for spk in spike_arr:
        count_per_time = np.sum(spk, axis=(0,1,2))
        max_cnt = np.max(count_per_time)
        max_counts.append(max_cnt)
        #print(spk.shape[:-1])
        ratio = max_cnt / np.prod(spk.shape[:-1])
        print ('Max spike ratio ', ratio)
    #max_cnt = max(max_counts)

    fig = plt.figure(tight_layout=True)
    for i, spk in enumerate(spike_arr):
        count_per_time = np.sum(spk, axis=(0,1,2))
        ax = plt.subplot(3, 1, i+1)
        plt.scatter(np.arange(count_per_time.shape[-1]), count_per_time, s=3, c='m')
        ax.set_title(titles[i])
        ax.set_ylim(bottom=0) #, top=max_cnt+10)
        ax.grid()
    plt.xlabel('time (ms)', fontsize=16)
    plt.ylabel('spike count', fontsize=16)
    plt.savefig(os.path.join(outdir, 'count_'+ layer_name+'.png'))
    plt.close()


def plot_spike_raster(spike_arr, name, layer_name=""):
    print('Plotting spike raster')
    for i in range(len(spike_arr)):
        print('total spikes ', np.sum(spike_arr[i]))
        if len(spike_arr[i].shape) == 1:
            spike_arr[i] = np.expand_dims(spike_arr[i], axis=(0,1,2))
        elif len(spike_arr[i].shape) == 2:
            spike_arr[i] = np.expand_dims(spike_arr[i], axis=(0,1))
        elif len(spike_arr[i].shape) == 3:
            spike_arr[i] = np.expand_dims(spike_arr[i], axis=(0))
        elif len(spike_arr[i].shape) == 4:
            pass
        else:
            print('Invalid data shape ', data.shape)
            return
    titles = ['aedat aggregated', 'poisson', 'aedat']

    C, H, W, T = spike_arr[0].shape
    N = H * W
    for c in range(C):
        fig = plt.figure(tight_layout=True)
        #st = fig.suptitle('Spike raster ' + layer_name + ': channel-'+str(c), fontsize=16)
        #fig.suptitle('Spike raster ' + layer_name + ': channel-'+str(c), y = 1.05, fontsize=16)
        plt.autoscale(False, axis='y')
        for i, spk in enumerate(spike_arr):
            mat = spk[c].reshape(-1, T) # N, T
            #print('mat size ', mat.shape)
            ax = plt.subplot(3, 1, i+1)
            for n in range(N):
                plt.plot((n+1)*mat[n,:] , '|b', markersize=3)
            ax.set_title(titles[i])
            plt.ylim(bottom=0.6, top=N)
        plt.xlabel('time (ms)', fontsize=16)
        plt.ylabel('neuron number', fontsize=16)
        plt.savefig(name+'_ch{}.png'.format(c)) #, bbox_extra_artists=[st], bbox_inches = 'tight')
        plt.close()
        break

plot(indir)
exit()

dirA = 'config2c_64_deep/batch256_r1z1_wrong'
dirB = 'config2c_64_deep/batch256_r2z1_wrong'
dirC = 'config2c_64_deep/batch256_r3z1_right'
dirD = 'config2c_64_deep/batch256_r3z1_interleave_wrong'
dirE = 'config2c_64_deep/batch256_r1z3_wrong'
outdir = 'config2c_64_deep'
batch_idx = 256

def getFile(dirname):
    all_files = os.listdir(dirname)
    for f in all_files:
        if f.endswith('.npz') and 'spike' in f:
            filename = os.path.join(dirname, f)
            return filename

def adjust_shape(tensor):
    shape = tensor.shape
    if len(shape) == 4:
        return tensor
    elif len(shape) == 3:
        tensor = np.expand_dims(tensor, axis=(0))
    elif len(shape) == 2:
        tensor = np.expand_dims(tensor, axis=(0, 1))
    else:
        print('Invalid shape ', shape)
    return tensor

def plot(a, b, c, outdir, filename):
    C,_,_,ta = a.shape
    C,_,_,tb = b.shape
    C,_,_,tc = c.shape
    for i in range(C):
        A = a[i]
        B = b[i]
        C = c[i]
        A = A.reshape(-1, ta)
        B = B.reshape(-1, tb)
        C = C.reshape(-1, tc)
        N,_ = A.shape
        print('channel {}: t(30): {}, t(60): {}, t(90): {}'.format(i, np.sum(A), np.sum(B), np.sum(C)))
        plt.figure()
        for n in range(N):
            plt.plot((n+1)*A[n,:], '|r', markersize=5)
            plt.plot((n+1)*B[n,:], '|g', markersize=5)
            plt.plot((n+1)*C[n,:], '|b', markersize=5)
        plt.ylim(bottom=0.7, top=N+1)
        plt.savefig(os.path.join(outdir, \
                'raster_L'+filename[9:]+'_bi{}_ch{}.png'.format(batch_idx, i)))
        plt.close()
        #break

def vary_time(dirA, dirB, dirC):
    dataC = np.load(getFile(dirC))
    dataB = np.load(getFile(dirB))
    dataA = np.load(getFile(dirA))
    for k, v in dataC.items(): # layers
        c = dataC[k]
        b = dataB[k]
        a = dataA[k]
        print('\n', k, ' c: {}, b: {}, a: {}'.format(c.shape, b.shape, a.shape))
        c = adjust_shape(c)
        b = adjust_shape(b)
        a = adjust_shape(a)
        print('#spikes: t(30): {}, t(60): {}, t(90): {}'.format(np.sum(a), np.sum(b), np.sum(c)))
        tc = c.shape[-1]
        tb = b.shape[-1]
        ta = a.shape[-1]
        b_pad = np.zeros((*c.shape[:-1], tc-tb))
        a_pad = np.zeros((*b.shape[:-1], tb-ta))
        a_new = np.concatenate((a, a_pad), axis=(-1))
        b_minus_a = b - a_new
        b_new = np.concatenate((b, b_pad), axis=(-1))
        c_minus_b = c - b_new
        plot(a, b_minus_a, c_minus_b, outdir, k)
        #break

def plot_interleaved(c, d, outdir, filename):
    C,_,_,tc = c.shape
    D,_,_,td = d.shape
    for i in range(C):
        C = c[i]
        D = d[i]
        C = C.reshape(-1, tc)
        D = D.reshape(-1, td)
        #print('channel {}, non_inter: {}, inter: {}'.format(i, np.sum(C), np.sum(D)))
        print('channel {}, repeated(3): {}, z-scaled: {}'.format(i, np.sum(C), np.sum(D)))
        N, _ = C.shape
        plt.figure()
        for n in range(N):
            plt.plot((n+1)*C[n,:], '|r', markersize=5, linewidth=5) #, label="full frame repeat"(red)
            plt.plot((n+1)*D[n,:], '|b', markersize=5, linewidth=5) #, label="interleaved frame repeat"(blue)
            #plt.plot((n+1)*C[n,:], '|r', markersize=5, linewidth=3) #, label="full frame repeat"(red)
            #plt.plot((n+1)*D[n,:], '|b', markersize=5, linewidth=5, alpha=0.7) #, label="interleaved frame repeat"(blue)
        plt.ylim(bottom=0.7, top=N+1)
        plt.savefig(os.path.join(outdir, \
                'z_L'+filename[9:]+'_bi{}_ch{}.png'.format(batch_idx, i)))
                #'interleave_L'+filename[9:]+'_bi{}_ch{}.png'.format(batch_idx, i)))
        plt.close()
        break

def vary_interleaving(dirC, dirD):
    dataC = np.load(getFile(dirC))
    dataD = np.load(getFile(dirD))
    for k, v in dataC.items(): # layers
        c = dataC[k]
        d = dataD[k]
        print(k, ' c: {}, d: {}'.format(c.shape, d.shape))
        c = adjust_shape(c)
        d = adjust_shape(d)
        print('non-interleaved: {}, interleaved: {}'.format(np.sum(c), np.sum(d)))
        plot_interleaved(c, d, outdir, k)

def vary_Z(dirC, dirE):
    dataC = np.load(getFile(dirC))
    dataE = np.load(getFile(dirE))
    for k, v in dataC.items(): # layers
        c = dataC[k]
        e = dataE[k]
        if 'input' in k:
            e /= 3
        print('\n', k, ' c: {}, e: {}'.format(c.shape, e.shape))
        c = adjust_shape(c)
        e = adjust_shape(e)
        print('repeated(3): {}, z-scaled: {}'.format(np.sum(c), np.sum(e)))
        tc = c.shape[-1]
        te = e.shape[-1]
        e_pad = np.zeros((*e.shape[:-1], tc-te))
        e_new = np.concatenate((e, e_pad), axis=(-1))
        plot_interleaved(c, e, outdir, k)

#vary_interleaving(dirC, dirD)
#vary_time(dirA, dirB, dirC)
vary_Z(dirC, dirE)
