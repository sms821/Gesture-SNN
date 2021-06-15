import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re, math

indir = 'config_400ms_2c_64'
layer_names = { 'input': 'input', 'spikerelu1': 'conv1', 'spikerelu3': 'conv2', \
        'spikerelu7': 'conv3', 'spikerelu9': 'conv4', 'spikerelu13': 'conv5'}
ann_layer_names = { 'input': 'input', 'relu1': 'conv1', 'relu2': 'conv2', \
        'relu4': 'conv3', 'relu5': 'conv4', 'relu7': 'conv5'}

raster_files = ['activations_pspike25.npz', 'activations_spike25.npz']

plain_corr_files = ['activations_2.npz', 'activations_spike2.npz']
comb4_corr_files = ['activations_0_comb4.npz', 'activations_spike0_comb4.npz']
comb6_corr_files = ['activations_0_comb6.npz', 'activations_spike0_comb6.npz']
combAll_corr_files = ['activations_0_comb-1.npz', 'activations_spike0_comb-1.npz']
full_aggr_files = ['activations_2.npz', 'activations_del__G-1_2_spike.npz']

spk_count_files = ['activations_spike2.npz', 'activations_del__G-1_2_spike.npz']

class AnalysisPlots:
    def __init__(self, root, raster_files):
        self.root         = root
        self.raster_files = raster_files
        self.outdir       = os.path.join(root, 'analysis_plots')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        self.n_plot_cols = len(layer_names.keys())

    def plotCorrelation(self, corr_files, color, outfile, type_):
        print('Plotting correlation...')
        all_files = [plain_corr_files, comb4_corr_files, combAll_corr_files, full_aggr_files]
        fig, axes = plt.subplots(len(all_files), self.n_plot_cols, sharex=False, gridspec_kw = {'wspace':0.00, 'hspace':0.05}, \
            figsize=(1.6*6, 1.4*len(all_files)))
        colors = [('r', 'm'), ('r', 'b'), ('r', 'g'), ('r', 'dimgrey')]
        types = ['(a) baseline', '(b) comb (1.6s)', '(c) comb (4s)', '(d) full aggr.']

        for AF, files in enumerate( all_files ):
            corr_files = files
            color = colors[AF]
            ann_data = []
            for row, file in enumerate(corr_files):
                print(file)
                col = 0
                filepath = os.path.join(self.root, file)
                data = np.load(filepath)
                for k, v in data.items():
                    #print(k, v.shape)
                    if k in layer_names.keys() or k in ann_layer_names.keys():
                        print(k, v.shape)
                        ann = v.flatten()
                        ann_data.append(ann)
                        if row == 0: # ann file
                            #annH = axes[AF, col].plot(ann, ann, color=color[row], label='relu', linewidth=1)
                            if AF == 0:
                                axes[AF, col].set_title(ann_layer_names[k], fontsize=12)

                        else:
                            #print('ann data: ', ann_data[col].shape)
                            spike_count = np.sum(v, -1)
                            spike_count = spike_count.flatten()
                            x_val = ann_data[col]
                            y_val = spike_count
                            annH = axes[AF, col].plot(np.unique(x_val), np.poly1d(np.polyfit(x_val, y_val, 1)) \
                                    (np.unique(x_val)), label='relu', linewidth=1, color='red')
                            snnH = axes[AF, col].scatter(ann_data[col], spike_count, s=1, color=color[row], label='spikerelu')

                        axes[AF, col].axes.xaxis.set_ticks([])
                        axes[AF, col].axes.yaxis.set_ticks([])
                        col += 1
                #break

            axes[AF, 0].legend(['ReLU', 'spikeReLU'], loc='upper left', prop={'size': 6})

        for i, ax in enumerate(axes.flat):
            #ax.set(ylabel=row_name[int(i / self.n_plot_cols)])
            if i % 6 == 0:
                ax.set(ylabel = types[int(i / 6)])
                #ax.set(ylabel='activated value\nann vs '+ types[int(i / 6)])
            ax.label_outer()
        fig.text(0.5, 0.07, 'ann pre-activation', ha='center', fontweight="bold", fontsize=14)
        fig.text(0.08, 0.5, 'activated value', va='center', rotation='vertical', fontweight="bold", fontsize=14)

        #handles, labels = axes[self.n_plot_cols-1].get_legend_handles_labels()
        #fig.legend(handles, labels, loc='center right')
        #plt.savefig(os.path.join(self.outdir, outfile), bbox_inches='tight')
        plt.savefig(os.path.join(self.outdir, 'correlation1.png'), bbox_inches='tight')
        plt.close()


    def plotRaster(self):
        print('Plotting rasters...')
        #colors = ['.brown', '.m']
        colors = ['tab:blue', '.m']
        fig, axes = plt.subplots(2, self.n_plot_cols, sharex=True, gridspec_kw = {'wspace':0.02, 'hspace':0}, \
                figsize=(7.2, 3.4))
        row_name = ['poisson', 'dvs']

        for row, file in enumerate( self.raster_files ):
            col = 0
            filepath = os.path.join(self.root, file)
            data = np.load(filepath)
            for k, v in data.items():
                if k in layer_names.keys():
                    print(k, v.shape)
                    self.plot_layer_spike_raster(axes[row, col], v, col, colors[row])
                    if row % 2 == 0:
                        axes[row, col].set_title(layer_names[k])
                    axes[row, col].sharey(axes[row-1, col])
                    axes[row, col].axes.xaxis.set_ticks([])
                    axes[row, col].axes.yaxis.set_ticks([])
                    col += 1
        fig.text(0.5, 0.04, 'time-steps', ha='center', fontweight='bold')
        fig.text(0.06, 0.5, 'neuron #', va='center', rotation='vertical', fontweight='bold')
        for i, ax in enumerate(axes.flat):
            ax.set(ylabel=row_name[int(i / self.n_plot_cols)])
            ax.label_outer()
        plt.savefig(os.path.join(self.outdir, 'raster.png'))
        #plt.savefig(os.path.join(self.outdir, 'raster.pdf'))
        plt.close()

    def plot_layer_spike_raster(self, axis, data, layer_num, color='|b'):
        C, H, W, T = data.shape
        numC = 4
        if layer_num == 0: # or layer_num == self.n_plot_cols-1:
            numC = C
        n_neurons = H * W * numC #min(750, H * W)

        mat = data[0:numC].reshape(-1, T)
        mat[mat == 0] = np.nan
        for n in range(n_neurons):
            axis.plot((n+1)*mat[n,:], color, markersize=0.9, marker='.')
        return n_neurons

    def plotSpkCount(self, spkCountFiles):
        print('Plotting spike count..')
        colors = ['m', 'dimgrey']
        R, C = 3, math.ceil( self.n_plot_cols / 3 )
        #fig, axes = plt.subplots(R, C, sharex=True, gridspec_kw = {'wspace':0.05, 'hspace':0}, \
        #        figsize=(6, 6))
        fig, axes = plt.subplots(R, C, sharex=True, figsize=(6, 4))
        row_name = ['dvs-baseline', 'dvs-aggregated']

        #for row, file in enumerate( self.raster_files ):
        for row, file in enumerate( spkCountFiles ):
            col = 0
            filepath = os.path.join(self.root, file)
            data = np.load(filepath)
            for k, v in data.items():
                if k in layer_names.keys():
                    print(k, v.shape)
                    count_per_time = np.sum(v, axis=(0,1,2))
                    axes[int(col / C), col % C].scatter(np.arange(count_per_time.shape[-1]), count_per_time, s=2, \
                            c=colors[row], label=row_name[row])
                    col += 1
        handles, labels = axes[R-1, C-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')
        plt.savefig(os.path.join(self.outdir, 'spikeCount.png'))
        plt.close()

def main():
    analysis = AnalysisPlots(indir, raster_files)
    analysis.plotCorrelation(full_aggr_files,    color=['r', 'dimgrey'],  outfile='corr_aggr.png',  type_='full aggr.')
    #analysis.plotCorrelation(combAll_corr_files, color=['r', 'gold'],   outfile='corr_comb-1.png',type_='comb (full clip)')
    #analysis.plotCorrelation(comb6_corr_files,   color=['r', 'g'],        outfile='corr_comb6.png', type_='comb (2.4s)' )
    #analysis.plotCorrelation(comb4_corr_files,   color=['r', 'b'],        outfile='corr_comb4.png', type_='comb (1.6s)' )
    #analysis.plotCorrelation(plain_corr_files,   color=['r', 'm'],        outfile='corr_plain.png', type_='baseline' )

    #analysis.plotRaster()
    #analysis.plotSpkCount(spk_count_files)

main()
exit()


indir = 'config_400ms_2c_64_deep'
indirs = ['config_800ms_2c_64_deep'] #, 'config_200ms_2c_64_deep', 'config_400ms_2c_64_deep', \
       #'config_600ms_2c_64_deep' ]
def modify_xlsx(indirs):
    import pandas as pd
    for dir in indirs:
        xldir = os.path.join(dir, 'xlsx')
        #all_files = os.listdir(dir)
        all_files = os.listdir(xldir)
        xl_files = ['combine_-1_spikerate.xlsx'] #, 'combine_-1_spikerate.xlsx']
        for file in all_files:
            #if file.endswith('.xlsx'):
            if file in xl_files:
                #xcel_file = os.path.join(dir, file)
                xcel_file = os.path.join(xldir, file)
                print('\n', xcel_file)
                df = pd.read_excel(xcel_file)
                data_np = df.to_numpy()
                #print(data_np)
                R, C = data_np.shape
                #print(R, C)
                lbl_count = data_np[R-1]
                wt_avg = np.zeros((R,1))
                for r in range(0, R-1):
                    wt_avg[r] = np.average(data_np[r,1:], weights=data_np[R-1,1:])
                #print(data_np[r, 0], wt_avg)
                df['wt_avg'] = pd.DataFrame(wt_avg)
                ann_ops = np.zeros((R,1))
                ann = np.loadtxt('ANN_Ops.txt', dtype=str)
                ann_ops[0:8,0] = ann[:,1].astype('float')
                snn_ops = np.multiply(wt_avg, ann_ops)
                ann_ops[-1,0] = np.sum(ann_ops)
                snn_ops[-1,0] = np.sum(snn_ops)
                #print(ann_ops, ann_ops.shape)
                #print(snn_ops, snn_ops.shape)
                df['ANN Ops'] = pd.DataFrame(ann_ops)
                df['SNN Ops'] = pd.DataFrame(snn_ops)
                print(df)
                #break
                df.to_excel( os.path.join(dir, 'updated_'+file) )


def plot(indir):
    all_files = os.listdir(indir)

    # collect spike file names
    spike_files = []
    for f in sorted( all_files ):
        if 'spike' in f and f.endswith('.npz'):
            spike_files.append(f)

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

def ann_activity(data_dict):
    print('Showing non-zero ratio in ANN..')
    for k, v in data_dict.items():
        nonzeros = np.nonzero(v)[0].shape
        tensor_sz = np.prod(v.shape)
        nz_ratio = nonzeros / tensor_sz
        print(k, nz_ratio)

def plot_aggr_type(indir):
    all_files = os.listdir(indir)

    # collect spike file names
    #spike_files = {'type1': [], 'type2': [], 'type3': []}
    #spike_files = {'type2': []}
    spike_files = {'type3': []}

    ann_files = []
    for f in sorted( all_files ):
        if f.endswith('.npz'):
            for k in spike_files.keys():
                if k in f:
                    spike_files[k].append(f)
                elif 'spike' in f and 'type' not in f:
                    spike_files[k].append(f)
            if 'spike' not in f:
                ann_activity(np.load( os.path.join(indir, f) ))

    for k, v in spike_files.items():
        print(k,v)
        # collect layer-wise keys
        data = np.load( os.path.join(indir, v[0]) )
        layer_names = list(data.keys())
        for L, ln in enumerate(layer_names):
            print('\nLayer ', ln)
            data_out = {}
            for filename in sorted(v):
                #if L == 0: print('Reading file ', filename)
                data_out[filename] = np.load(os.path.join(indir, filename))[ln]
            overlay_spike_count(data_out, indir+'/aggr_types', k+'_count_'+ln)
            #break
        #break

def overlay_spike_count(data: dict, outdir: str, fileprefix: str):
    other_colors = ['darkorchid', 'magenta', 'blue', 'darkgoldenrod',\
            'darkorange', 'saddlebrown', 'teal', 'gray', 'deeppink', \
            'indianred', 'olive', 'springgreen', 'dodgerblue', 'navy']
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print('Plotting spike counts..')
    legends = []
    fig = plt.figure(tight_layout=True)
    for z, (k, v) in enumerate(data.items()):
        x = re.findall("(G(?:-|[0-9])[0-9]*)", k)
        xx = re.findall("K[0-9][0-9]*", k)
        #x = re.findall("((?:G-|K[0-9])[0-9]*)", k)
        if len(x) == 0:
            if 'pspike' in k:
                legends.append('poisson')
                color = 'darkred'
            else:
                legends.append('plain')
                color = 'darkgreen'
        else:
            if '-1' in k:
                legends.append('full aggr')
                color = 'black'
            else:
                #legends.append('G: '+x[0][1:])
                legends.append('K: '+xx[0][1:])
                color = other_colors[z]
                y = re.findall("(G(?:-|[0-9])[0-9]*)", k)
                #print(y)

        num_dims = len(v.shape)
        rem_dims = []
        for i in range(0, 4-num_dims):
            rem_dims.append(i)
        data = v
        if len(rem_dims) > 0:
            data = np.expand_dims(v, axis=(rem_dims))
        count_per_time = np.sum(data, axis=(0,1,2))
        max_cnt = np.max(count_per_time)
        ratio = max_cnt / np.prod(data.shape[:-1])
        print(k, 'total spikes: ', np.sum(v), 'max-spike-ratio: {:.4f}'.format(ratio))
        #plt.scatter(np.arange(count_per_time.shape[-1]), count_per_time, s=3)
        plt.plot(count_per_time)
        #plt.plot(count_per_time[0:52]) #, color=color)
    plt.title('G = ' + y[0][1:])
    plt.xlabel('time (ms)', fontsize=16)
    plt.ylabel('spike count', fontsize=16)
    plt.legend(legends)
    #plt.savefig(os.path.join(outdir, fileprefix+'_G'+y[0][1:]+'_K1,2,5,10,15'))
    #plt.savefig(os.path.join(outdir, fileprefix+'_G'+y[0][1:]+'_K1,2,5,10,15_zoom'))
    plt.savefig(os.path.join(outdir, fileprefix+'_G'+y[0][1:]+'_K25,35,50,100'))
    plt.close()

def compare_activations(indir):
    #filenames = ['activations_25.npz', 'activations_spike25.npz', 'activations_del__G-1_25_spike.npz']
    #filenames = ['activations_spike25.npz', 'activations_del__G-1_25_spike.npz']
    #filenames = ['activations_2.npz', 'activations_spike2.npz', 'activations_del__G-1_2_spike.npz']
    #filenames = ['activations_2.npz', 'activations_spike2.npz']
    filenames = ['activations_2.npz', 'activations_del__G-1_2_spike.npz']
    #filenames = ['activations_2.npz', 'activations_pspike2.npz']
    #filenames = ['activations_spike2.npz', 'activations_del__G-1_2_spike.npz']

    data_dict = {}
    #legends = ['ann', 'aedat', 'snn_full_aggr']
    #legends = ['ann', 'poisson', 'snn_full_aggr']
    legends = ['ann', 'aedat_full_aggr']
    start = 0
    max_acts = np.loadtxt(os.path.join(indir, 'max_acts_99.7.txt'))
    for i in range(start, len(filenames)):
        data_file = os.path.join(indir, filenames[i])
        data = np.load(data_file)
        #print('snn: ', list(snn.keys()))
        for n, (k,v) in enumerate( data.items() ):
            v = data[k]
            if 'layer-'+str(n) not in data_dict.keys():
                data_dict['layer-'+str(n)] = [v]
            else:
                data_dict['layer-'+str(n)].append(v)
            #print(k, v.shape)
    overlay_activations(data_dict, indir, legends[start:], max_acts)

def adjust_dims(inp, out):
    num_dims = len(inp.shape)
    rem_dims = []
    for i in range(0, 4-num_dims):
        rem_dims.append(i)
    out = inp
    if len(rem_dims) > 0:
        out = np.expand_dims(inp, axis=(rem_dims))

def overlay_activations(data: dict, outdir: str, legends: list, max_acts=[]):
    for layer_num, (k, v) in enumerate(data.items()): # spanning layers
        print('\n{}'.format(k))

        #print('ann: ({:.4f}, {:.4f}), mean: {:.4f}'.format(ann.min(), ann.max(), ann.mean()) )
        plt.figure()
        '''
        for i in range(0, len(v)):
            T = v[i].shape[-1]
            if len(v[i].shape) == 4: #snn non fc layers
                tensor = np.sum(v[i], -1)
                tensor /= T
                tensor *= 100
            elif len(v[i].shape) == 2: #snn fc layers
                tensor = v[i][:,-1]
                tensor /= T
                tensor *= 100
            else:
                tensor = v[i]
            tensor = tensor.flatten()
            print('{}, range: ({:.4f}, {:.4f}), mean: {:.4f}'.format(legends[i], tensor.min(), \
                    tensor.max(), tensor.mean() ))
            plt.hist(tensor, bins='auto', alpha=0.6) #, density=True)
        plt.legend(legends)
        plt.savefig(os.path.join(outdir, 'hist25_{}.png'.format(k)))
        '''

        ann = v[0]
        for i in range(1, len(v)):
            snn = v[i]
            #print(snn.shape)
            T = snn.shape[-1]
            if len(snn.shape) == 2:
                spike_count = snn[:,-1]
            else:
                spike_count = np.sum(snn, -1)
            print('snn: ({:.4f}, {:.4f}), mean: {:.4f}, total: {}'.format(spike_count.min(), \
                    spike_count.max(), spike_count.mean(), np.sum(snn)) )
            #print()
            assert spike_count.shape == ann.shape, 'snn: {}, ann: {}'.format(spike_count.shape, \
                    ann.shape)
            ann = ann.flatten()
            spike_count = spike_count.flatten()
            max_ann_pixel = np.max(ann)
            norm_ann = ann / max_ann_pixel

            # poisson rate
            #spike_rate = spike_count / snn.shape[-1]
            #plt.plot(norm_ann, norm_ann, color='red')
            #plt.scatter(norm_ann, spike_rate, s=2, color='blue')

            # plain, poisson count, full aggr.
            plt.plot(ann, ann, color='red')
            plt.scatter(ann, spike_count, s=2, color='blue')

        plt.legend(legends)
        plt.title(k)

        # poisson rate
        #plt.ylabel('poisson spike rate')
        #plt.savefig(os.path.join(outdir, 'poisson_rate2_{}.png'.format(k)))
        #plt.xlabel('ann activation ratio (a/amax)')

        # poisson count
        #plt.xlabel('ann activation (a)')
        #plt.ylabel('poisson spike count')
        #plt.savefig(os.path.join(outdir, 'poisson_count2_{}.png'.format(k)))

        # aggr
        plt.xlabel('ann activation (a)')
        plt.ylabel('full aggr. spike count')
        plt.savefig(os.path.join(outdir, 'full_aggr_count2_{}.png'.format(k)))

        ## plain
        #plt.xlabel('ann activation (a)')
        #plt.ylabel('aedat spike count')
        #plt.savefig(os.path.join(outdir, 'plain_count2_{}.png'.format(k)))

        #plt.savefig(os.path.join(outdir, 'hist25_{}.png'.format(k)))
        #plt.savefig(os.path.join(outdir, 'scatter2_{}.png'.format(k)))
        #plt.savefig(os.path.join(outdir, 'scatter_plain2_{}.png'.format(k)))





#plot_aggr_type(indir)
#plot(indir)
#compare_activations(indir)
#modify_xlsx(indirs)
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
