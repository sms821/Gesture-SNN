import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import numpy as np
import copy
from PIL import Image
import pandas as pd
from more_itertools import unique_everseen
import spikeFileIO as io
from datetime import datetime
from collections import OrderedDict
import yaml
import pprint

# Consider dictionary for easier iteration and better scalability
class yamlParams(object):
    '''
    This class reads yaml parameter file and allows dictionary like access to the members.
    '''

    def __init__(self, parameter_file_path):
        with open(parameter_file_path, 'r') as param_file:
            self.parameters = yaml.safe_load(param_file)
        obj = pprint.PrettyPrinter(indent=0)
        obj.pprint(self.parameters)

    # Allow dictionary like access
    def __getitem__(self, key):
        return self.parameters[key]

    def __setitem__(self, key, value):
        self.parameters[key] = value

    def save(self, filename):
        with open(filename, 'w') as f:
            yaml.dump(self.parameters, f)


class AedatConverter:
    """
    sample_file: .txt file containing input filename and label
    root: directory where files listed in sample_file are located
    accumulation_method: `spikecount` -- constant #events per frame
                         `timeinterval` -- constant interval per frame
    accumulation_value: is accumulation_method == `spikecount`: #events per frame\
            else: frame duration in granularity of `sampling_time`.
    output_resolution: (X,Y) dimension of generated data
    keep_polarity: `True/False` -- single or dual-channel
    sampling_time: event sampling granularity in us. powers of 10
    video_duration: sample duration of the video from which to create \
            frames in seconds. If < 0, use the whole clip
    """

    def __init__(
        self,
        sample_file: str,
        root: str,
        accumulation_method: str = "spikecount",
        accumulation_value: float = 3000,
        output_resolution: tuple = (64, 64),
        keep_polarity: bool = False,
        sampling_time: int = 1000,
        video_duration: float = 1.5,
        start_offset: int = 0,
        only_labels: int = None,
        num_ch_per_frame: int = None,
        overlap_factor: int = None,
        samples_only: bool = True,
    ):
        if accumulation_method not in ["timeinterval", "spikecount"]:
            raise ValueError(f"accumulation_method must be one of" " 'timeinterval', 'spikecount'")
        if accumulation_method and not accumulation_value:
            raise ValueError(f"accumulation_value cannot be zero")
        self.sample_file = sample_file
        self.root = root
        self.accumulation_method = accumulation_method
        self.accumulation_value = accumulation_value
        self.H, self.W = output_resolution
        self.keep_polarity = keep_polarity
        self.sampling_time = sampling_time
        self.video_duration = video_duration
        self.frames_per_class = {}
        self.start_offset = start_offset
        self.only_labels = only_labels
        # for creating time parallel frames
        self.num_ch_per_frame = num_ch_per_frame
        self.overlap_factor = overlap_factor
        self.samples_only = samples_only

        # read the sample file
        self.file_list = []
        self.label_list = []
        samples = np.loadtxt(sample_file).astype('int')
        for s in samples:
            input_index = s[0]
            class_label = s[1]
            filename = os.path.join(root, str(input_index)+'.npy')
            self.file_list.append(filename)
            self.label_list.append(class_label)
        assert len(self.file_list) > 0
        assert len(self.label_list) > 0

        C = 1
        if self.keep_polarity:
            C = 2
        self.frame_dim = (C, self.H, self.W)
        print('///////////// Dataset configs /////////////')
        print('acc. method:', self.accumulation_method, \
                'acc. value:', self.accumulation_value)
        print('frame res:', output_resolution, 'polarity:', keep_polarity)
        print('sampling time:', sampling_time, 'start offset:', start_offset)
        print('only labels:', only_labels)
        print('num ch. per frame: ', num_ch_per_frame)
        print('num overlapping channels: ', overlap_factor)
        print('//////////////////////////////////////////')

    def update_frame_count(self, num_frames):
        if self.label not in self.frames_per_class.keys():
            self.frames_per_class[self.label] = num_frames
        else:
            self.frames_per_class[self.label] += num_frames

    def _accumulate_by_count(self, xypt):
        """ xypt: y(0),x(1),p(2),t(3) shape: 4,N """

        # prepare bins so that they are common to all frames
        n_spk = xypt.shape[-1]
        print('num spikes: ', n_spk)

        # get rid of incomplete frames
        incomplete = n_spk % self.accumulation_value
        if incomplete:
            xypt = xypt[:, :-incomplete]

        # group spike trains by frame they will end up in
        sliced_xypt = np.split(xypt, range(self.accumulation_value, n_spk, self.accumulation_value), axis=1)
        sliced_xypt = sliced_xypt[:-1] # 0,1,2,3 == y,x,p,t
        num_frames = len(sliced_xypt)
        print('num frames: ', num_frames)

        # every frame gets computed from the spiketrain as a histogram
        if self.keep_polarity:
            frames = np.empty((num_frames, self.H, self.W, 2), dtype=np.uint16)
            for i, slice_item in enumerate(sliced_xypt):
                # slice_item: x,y,p,t

                # verify the spike times
                TD_slice = io.event(xEvent=slice_item[1,:], yEvent=slice_item[0,:], \
                        pEvent=slice_item[2,:], tEvent=slice_item[3,:])
                #print('checking frame events..')
                self.check_spikes(TD_slice)

                frames[i] = np.histogramdd(( slice_item[0,:], slice_item[1,:], slice_item[2,:] ),
                                            bins=(*self.bins_yx, (-1, 0.5, 2) ))[0]
            frames = np.transpose(frames, (0,3,1,2))
            #print('frames: ', frames.shape)
        else:
            frames = np.empty((num_frames, self.H, self.W), dtype=np.uint16)
            for i, slice_item in enumerate(sliced_xypt):
                #print('slice_item ', slice_item.shape)

                # verify the spike times
                TD_slice = io.event(xEvent=slice_item[1,:], yEvent=slice_item[0,:], \
                        pEvent=slice_item[2,:], tEvent=slice_item[3,:])
                #print('checking frame {} events..'.format(i))
                self.check_spikes(TD_slice)

                # how many unique time stamps in this frame?
                unq = set(TD_slice.t)
                if (len(unq) > 300):
                    print('[WARNING] # unique timestamps: ', len(unq))

                frames[i] = np.histogram2d(slice_item[0,:], slice_item[1,:], bins=self.bins_yx)[0]
            frames = np.expand_dims(frames, axis=1)

        # just some sanity checks
        assert frames.sum() == sum(sl.shape[1] for sl in sliced_xypt)
        #print('event count ', frames.sum() / num_frames)

        # update frame count for class
        self.update_frame_count(num_frames)
        print('frames: ', frames.shape)
        print('spike_frames: ')
        for i, s in enumerate(sliced_xypt):
            print(i, s.shape)

        return frames, sliced_xypt


    def _accumulate_by_time(self, xypt):
        """ xypt: y(0),x(1),p(2),t(3) shape: 4,N """

        x, y, p, t = xypt[1,:], xypt[0,:], xypt[2,:], xypt[3,:]
        timebins = np.arange(t[0], t[-1]+2, self.accumulation_value)
        duration = t[-1] - t[0] + 1 # in ms
        incomplete = duration % self.accumulation_value
        if incomplete:
            timebins = timebins[:-1]
        num_frames = len(timebins) - 1

        # accumulate and spatially bin
        if self.keep_polarity:
            frames, _ = np.histogramdd((t, p, y, x), (timebins, (-1, 0.5, 2), \
                    *self.bins_yx ))
        else:
            frames, _ = np.histogramdd((t, y, x), (timebins, *self.bins_yx))
            frames = np.expand_dims(frames, axis=1)

        sliced_xypt = []
        spk_per_frame = frames.sum(axis=(1, 2, 3)).astype(int)
        sliced_xypt = np.empty(len(spk_per_frame), dtype=np.ndarray)
        running_spk = 0
        for i, n in enumerate(spk_per_frame):
            sliced_xypt[i] = xypt[:,running_spk: (running_spk + n)]
            running_spk += n

        ## just some sanity checks
        assert frames.sum() == sum(sl.shape[-1] for sl in sliced_xypt)

        # update frame count for class
        self.update_frame_count(num_frames)

        return frames, sliced_xypt


    def accumulate(self, TD):
        """
        Perform accumulation based on the chosen method and parameter.

        :param TD: Numpy array, (4, N), with TD.x, TD.y, TD.t, TD.p values of each spike.
        :returns: (frames, sliced_xyt); `frames` is an array with the spike \
        raster, shaped (n time bins, resolution, resolution). `sliced_xyt` is \
        a list of spike trains, of the same form as `xyt`, one for each frame.
        """
        #print('time range min, max: ', TD.t.min(), TD.t.max(), '1st, last: ', TD.t[0], TD.t[-1])
        # compute spatial grid
        x, y = TD.x, TD.y
        bins_x = np.linspace(x.min(), x.max(), self.W + 1)
        bins_y = np.linspace(y.min(), y.max(), self.H + 1)
        self.bins_yx = (bins_y, bins_x)

        xypt = TD.xypt[:,self.start_offset:]
        #print('xypt time range min, max: ', xypt[3,:]min(), TD.t.max(), '1st, last: ', TD.t[0], TD.t[-1])
        #print('xypt: ', xypt.shape)

        if self.accumulation_method == "spikecount":
            return self._accumulate_by_count(xypt)
        if self.accumulation_method == "timeinterval":
            return self._accumulate_by_time(xypt)
        raise ValueError("accumulation_method must be either 'spikecount' or 'timeinterval'.")


    def check_spikes(self, TD):
        #print('Checking spikes..')
        tmin, tmax = TD.t.min(), TD.t.max()
        if not tmin == TD.t[0]:
            print('video clip: tmin: {}  t[0]: {}'.format(tmin, TD.t[0]))
            print('stopping data generation')
            exit()
        if not tmax == TD.t[-1]:
            print('video clip: tmax: {}  t[-1]: {}'.format(tmax, TD.t[-1]))
            print('stopping data generation')
            exit()
        return tmin, tmax


    def read_aedat_bin_file(self, in_file: str):
        """
        Read and return a single aedat file.

        :param in_file: The path to the file.
        :returns: (frames, spiketrain), respectively accumulated frames \
        and a (4, N) array of (x, y, p, t) event coordinates.
        """
        # read the .bs2 version of aedat file
        filepath = os.path.join(self.root, in_file)

        # read the .npy version of aedat file
        TD = io.readNpSpikes(filepath) # NOTE: the time stamps are in ms here

        # clipping video according to required duration
        duration = (TD.t[-1] - TD.t[0]) # in ms
        rqd_duration = min(duration, self.video_duration*1000) # in ms
        if self.video_duration < 0:
            rqd_duration = duration
        print('Duration original: {} ms, required: {} ms'.format( duration, rqd_duration ))

        if self.sampling_time == 1000: # ms-sampling
            time_axis_dim = int(rqd_duration)
            spikeTensor = TD.toSpikeArray() # ensures events are ms-sampled
            #dim = (*self.frame_dim, time_axis_dim)
            TD_subsampled = io.spikeArrayToEvent(spikeTensor[:,:,:,0:time_axis_dim]) # re-orders events
            xypt = TD_subsampled.xypt
            xypt = xypt[:,np.argsort(xypt[3,:])]
            TD_subsampled.x = xypt[1,:]
            TD_subsampled.y = xypt[0,:]
            TD_subsampled.p = xypt[2,:]
            TD_subsampled.t = xypt[3,:]
            TD_subsampled.xypt = xypt

        else:
            time_range = np.argwhere( TD.t <= rqd_duration )
            range_index = int( time_range[-1][0] )
            TD_subsampled = io.event( TD.x[0:range_index], TD.y[0:range_index], TD.p[0:range_index], \
                    TD.t[0:range_index] )

        print('checking video events..')
        self.check_spikes(TD_subsampled)

        # accumulate and return
        frames, slicedTD = self.accumulate(TD_subsampled)
        return frames, slicedTD


    def save_images(self, outdir, filename, id_):
        arr = np.load(os.path.join(outdir, filename))
        spiketrain = arr['spiketrain']
        frame = arr['frame']
        #print('frame ', frame.shape, ' spiketrain ', spiketrain.shape)
        assert np.sum(frame) == spiketrain.shape[1], 'frame: {}, spiketrain: {}'.format(np.sum(frame), spiketrain.shape[1])
        y = spiketrain[0,:]
        x = spiketrain[1,:]
        p = spiketrain[2,:]
        t = spiketrain[3,:]
        #TD = io.event(x, y, p, t)
        if self.keep_polarity:
            spike_frame = np.empty((1, self.H, self.W, 2)) #, dtype=np.uint16)
            # slice_item: x,y,p,t
            spike_frame[0] = np.histogramdd((y, x, p), bins=(*self.bins_yx, (-1, 0.5, 2)))[0]
            spike_frame = np.transpose(spike_frame, (0, 3, 1, 2))
        else:
            spike_frame = np.empty((1, self.H, self.W)) #, dtype=np.uint16)
            spike_frame[0] = np.histogram2d(y, x, bins=self.bins_yx)[0]
            spike_frame = np.expand_dims(spike_frame, axis=0)

        assert np.sum(frame) == np.sum(spike_frame)
        #print('spike frame ', spike_frame.shape)

        frame = np.expand_dims(frame, axis=0)
        frame_max = max(1, frame.max()).astype(frame.dtype)
        spike_frame_max = max(1, spike_frame.max()).astype(spike_frame.dtype)
        frame = frame / frame_max
        spike_frame /= spike_frame_max
        visualize_as_image(spike_frame, outdir, filename+'_spike_{}'.format(id_), normed=True)
        visualize_as_image(frame, outdir, filename+'_nospike_{}'.format(id_), normed=True)


    def to_folder(
        self, out_dir: str, overwrite: bool = False, compressed: bool = False,

    ):
        import datetime

        if 'train' in self.sample_file:
            prefix = 'train'
        else:
            prefix = 'test'

        # create the folders
        os.makedirs(out_dir+'/'+prefix, exist_ok=True)
        if not overwrite and os.listdir(out_dir):
            raise Exception("Directory not empty. Set overwrite=True to ignore")
        savez = np.savez_compressed if compressed else np.savez

        postfix = ''
        if self.start_offset is not None and self.start_offset > 0:
            postfix = '_off{:.1f}k'.format(self.start_offset/1000)

        labels = set(self.label_list)
        label_to_filename = {}
        for l in labels:
            label_to_filename[l] = []

        # go through files and save them
        for file_i in range(len(self.file_list)):
            file, label = self.file_list[file_i], self.label_list[file_i]
            if self.only_labels is not None and not label == self.only_labels:
                continue
            self.label = label
            print('\nProcessing aedat file ', file)
            frames, spiketrains = self.read_aedat_bin_file(file)
            #print('spiketrains ', len(spiketrains))

            for i, frame in enumerate(frames):
                fileprefix = '{}_{}'.format(file_i, i) + postfix
                #spaceStr = ' '*(18 - len(fileprefix))
                #writeStr = fileprefix + spaceStr + str(label) + '\n'
                #labelFile.write(writeStr)
                fname = fileprefix + '.npz'
                label_to_filename[label].append(fileprefix)

                #print(spiketrains[i].shape, 'len ', len(spiketrains[i]))
                savez(
                    os.path.join(out_dir+'/'+prefix, fname),
                    frame=frame,
                    spiketrain=spiketrains[i],
                    label=label,
                    original_filename=file,
                    bins_yx=self.bins_yx,
                )
                if self.samples_only:
                    self.save_images(out_dir+'/'+prefix, fname, id_=i) # to verify
            if self.samples_only:
                break
            #if file_i > 20:
            #    break

        # create the label file
        lfilename = prefix + '.txt'
        lfilepath = os.path.join(out_dir, lfilename)
        labelFile = open(lfilepath, 'a')
        if os.path.getsize(lfilepath) == 0:
            labelFile.write('#sample       #class\n')
        else:
            print('[WARNING] label file not empty.')
            labelFile = open( lfilepath+'_backup_'+str(datetime.datetime.now()), 'a' )
            labelFile.write('#sample       #class\n')

        sorted_labels = sorted(label_to_filename.keys())
        #for label, files in label_to_filename.items():
        for i in range(len(sorted_labels)):
            label = sorted_labels[i]
            files = label_to_filename[label]
            for f in files:
                spaceStr = ' '*(18 - len(f))
                writeStr = f + spaceStr + str(label) + '\n'
                labelFile.write(writeStr)
        labelFile.close()

        # print num frames per class
        print('frames per class:')
        for k,v in self.frames_per_class.items():
            print(k, v)


    def save_3c_images(self, outdir, filename):
        arr = np.load(os.path.join(outdir, filename), allow_pickle=True)
        frame = arr['frame']
        c, h, w = frame.shape
        #print('frame shape ', frame.shape)
        spiketrain = arr['spiketrain'].item()
        bins_yx = arr['bins_yx']
        #print('bins_yx ', bins_yx.shape)
        #print(spiketrain.keys())
        spike_frames = np.zeros((c, h, w))
        for i, (k, v) in enumerate(spiketrain.items()):
            #print(k, v.shape)
            #continue
            assert np.sum(frame[i]) == v.shape[1], 'frame{}: {}, spiketrain: {}'.format(i, \
                    np.sum(frame[i]), spiketrain.shape[1])
            y = v[0,:]
            x = v[1,:]
            p = v[2,:]
            t = v[3,:]
            spike_frame = np.empty((1, self.H, self.W))
            spike_frame[0] = np.histogram2d(y, x, bins=bins_yx[i,...])[0]
            spike_frames[i] = spike_frame

        frame = np.expand_dims(frame, axis=0)
        frame_max = max(1, frame.max()).astype(frame.dtype)
        frame = frame / frame_max
        spike_frames = np.expand_dims(spike_frames, axis=0)
        spike_frame_max = max(1, spike_frames.max()).astype(spike_frames.dtype)
        spike_frames = spike_frames / spike_frame_max
        visualize_as_image(frame, outdir, filename[:-4]+'fr', normed=True)
        visualize_as_image(spike_frames, outdir, filename[:-4]+'_spk', normed=True)

    def create_time_parallel_frames(
            self, in_dir: str, out_dir: str, overwrite: bool = False, compressed: bool = False,

    ):
        print('indir: ', in_dir)
        src_dataset_dir = in_dir.split('/')[-1]
        out_dir = os.path.join(out_dir, 'tp_'+src_dataset_dir)
        if 'train' in self.sample_file:
            prefix = 'train'
        else:
            prefix = 'test'

        # create the folders
        os.makedirs(out_dir+'/'+prefix, exist_ok=True)
        if not overwrite and os.listdir(out_dir):
            raise Exception("Directory not empty. Set overwrite=True to ignore")
        savez = np.savez_compressed if compressed else np.savez

        lfilename = prefix + '.txt'
        lfilepath = os.path.join(in_dir, lfilename)
        print('label file: ', lfilepath)
        labelFile = open(lfilepath, 'r')
        all_files = np.loadtxt(labelFile, dtype=str)
        file_names = list(all_files[:,0])
        labels     = list(all_files[:,1])
        paired     = list(zip(file_names, labels))
        def sortByLabel(e):
            return int(e[1])
        paired.sort(key=sortByLabel)
        #print('sorted pairs')
        #for a,b in paired:
        #    print(a, b)

        # create a dict from label to filenames
        label_to_filename = {}
        for a, b in paired:
            label_to_filename[b] = []
        for a, b in paired:
            label_to_filename[b].append(a)
        #print('label to filename dict: ')
        #for k,v in label_to_filename.items():
        #    print(k, v)

        num_frames_org = len(paired) #NOTE: input frames should be single channeled

        num_ch_per_frame = self.num_ch_per_frame
        overlap_factor   = self.overlap_factor

        stride = num_ch_per_frame - overlap_factor

        # create a label file for new dataset
        import datetime
        lfilename = prefix + '.txt'
        lfilepath = os.path.join(out_dir, lfilename)
        labelFile = open(lfilepath, 'a')
        if os.path.getsize(lfilepath) == 0:
            labelFile.write('#sample       #class\n')
        else:
            print('[WARNING] label file not empty.')
            labelFile = open( lfilepath+'_backup_'+str(datetime.datetime.now()), 'a' )
            labelFile.write('#sample       #class\n')

        outfiles_dict = {}
        for k in label_to_filename.keys():
            outfiles_dict[k] = 0
        # processing all frames class-wise (start)
        for k, v in label_to_filename.items():
            print('===== label {} ====='.format(k) )
            num_frames_org = len(v)
            src_files = v # list of files for current class
            #print('filenames: ')
            #print(v)

            # number of time parallel frames
            num_frames_tp = int((num_frames_org - num_ch_per_frame) / stride)
            #outfiles_dict[k] = num_frames_tp

            from collections import deque
            #curr_frame_files = []
            curr_frame_files = deque()
            for ovf in range(overlap_factor):
                curr_frame_files.append(v[ovf])

            # process all the time parallel frames in this class (start)
            for nf in range(num_frames_tp):

                # add new files for current frame
                #print('src_file_idx: ')
                #f_idx = []
                for s in range(stride):
                    fi = nf*stride + overlap_factor + s
                    #print(fi)
                    curr_frame_files.append(v[fi])
                    #f_idx.append(fi)
                #print(f_idx)
                assert len(curr_frame_files) == num_ch_per_frame, \
                        'len(curr_frame_files): {} and num_ch_per_frame: {} do not match'.format(curr_frame_files, num_ch_per_frame)
                # curr_frame_files now has all files needed to create new frame

                print('frame {} files {}'.format(nf, curr_frame_files))
                # collect numpy arrays from files (start)
                for i, ccf in enumerate(curr_frame_files):
                    #print(ccf)
                    file = os.path.join(in_dir, prefix, ccf+'.npz')
                    with np.load(file) as f:
                        frame = f['frame'].astype(np.float32)
                        spiketrain = f['spiketrain'].astype(np.float32)
                        bins_yx = f['bins_yx']

                        if i == 0:
                            frame_shape = frame.shape
                            assert frame_shape[0] == 1, 'frame shape {} should be single-channeled'.format(frame_shape)
                            spktrain_shape = spiketrain.shape
                            bins_yx_shape = bins_yx.shape
                            frames = np.zeros((num_ch_per_frame, *frame_shape[1:]))
                            bins_yx_all = np.zeros((num_ch_per_frame, \
                                    *bins_yx_shape))
                            spiketrains = {}

                        # merge the frames and spike trains
                        frames[i,...] = np.squeeze(frame, axis=0)
                        spiketrains[str(i)] = spiketrain
                        bins_yx_all[i,...] = bins_yx
                # collect numpy arrays from files (end)

                # save filename and label to label file
                fileprefix = str(nf) + '_' + str(k)
                spaceStr = ' '*(18 - len(fileprefix))
                label = k
                writeStr = fileprefix + spaceStr + str(label) + '\n'
                labelFile.write(writeStr)

                fname = fileprefix + '.npz'
                ## save frames and spiketrains to npz
                #savez(
                #    os.path.join(out_dir+'/'+prefix, fname),
                #    frame=frames,
                #    spiketrain=spiketrains,
                #    label=k,
                #    bins_yx=bins_yx_all,
                #)
                outfiles_dict[k] += 1

                # create a 3-channel image and verify
                self.save_3c_images(out_dir+'/'+prefix, fname)
                break

                # remove files not needed in next frame
                for s in range(stride):
                    curr_frame_files.popleft()
                #if nf == 99:
                #    break
            # process all the time parallel frames in this class (end)
            #break
        # processing all frames class-wise (end)
        labelFile.close()
        print('num files written per class')
        for k, v in outfiles_dict.items():
            print(k, v)



def uniqueEventCount(arr):
    ''' identify unique events, arr shape = N, 4 '''

    N, four = arr.shape
    assert four == 4, "Given arr: {}, Required shape: N, 4".format(arr.shape)
    all_events = []
    for i in range(N):
        all_events.append((arr[i, 0], arr[i, 1], arr[i, 2], arr[i, 3]))
    unique_events = list(unique_everseen(all_events))
    print('Number of unique events: ', len(unique_events))
    return unique_events

def toCompactSpikeArray(TD, dim=(2,128,128)):
    ''' Converts TD events to a dense array, maintains the
    time ordering of spikes but compacts them in time '''

    num_events = len(TD.x)
    #print('original num events in TD ', num_events)
    all_events = np.zeros((num_events, 4), dtype=TD.x.dtype)
    all_events[:,0], all_events[:,1], all_events[:,2], all_events[:,3] = TD.x, TD.y, TD.p, TD.t
    unique_timestamps = list(unique_everseen(TD.t))
    nut = len(unique_timestamps) # #timesteps
    #print('time steps: ', nut)

    #print('** TD events **')
    #unique_events = uniqueEventCount(all_events)

    all_events_sorted = all_events[np.argsort(all_events[:,3])]
    #print('all events sorted: ', all_events_sorted.shape)

    spikeArr = np.zeros((2,128,128,nut))
    x,y,p,t = all_events_sorted[0,:]
    jt = 0
    spikeArr[p, y, x, jt] = 1
    for i in range(1, num_events):
        x, y, p, t = all_events_sorted[i,:]
        if all_events_sorted[i,3] != all_events_sorted[i-1,3]:
            jt += 1
        spikeArr[p, y, x, jt] = 1
    return spikeArr


class LambdaBase(nn.Sequential):
     def __init__(self, fn, *args):
         super(LambdaBase, self).__init__(*args)
         self.lambda_func = fn

     def forward_prepare(self, input):
         output = []
         for module in self._modules.values():
             output.append(module(input))
         return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class Stats:
    ''' Class for saving and restoring learning stats '''

    def __init__(self):
        self.best_train_acc = 0
        self.best_train_loss = math.inf
        self.best_test_acc = 0
        self.best_test_loss = math.inf
        self.epochs = []
        self.train_epoch_loss = []
        self.train_epoch_acc = []
        self.test_epoch_loss = []
        self.test_epoch_acc = []

class Solver:
    """ Class for training and testing the model """

    def __init__(self, **kwargs):
        self.net = kwargs['net']
        self.device = kwargs['device']
        self.trainloader = kwargs.get('trainloader', None)
        self.testloader = kwargs.get('testloader', None)
        self.optimizer = kwargs.get('optimizer', None)
        self.criterion = kwargs.get('criterion', None)
        self.lr = kwargs.get('lr', None)
        self.chkpt_dir = kwargs.get('chkpt_dir', None)
        self.chkpt_filename = kwargs.get('chkpt_filename', None)
        self.num_epochs = kwargs.get('num_epochs', None)
        self.resume = kwargs.get('resume', None)
        self.checkpoint = kwargs.get('save_checkpoint', None)
        self.num_classes = kwargs.get('num_classes', None)
        self.interval_display = kwargs['interval_display']
        self.criterion_test = kwargs['criterion_test']

        self.stats = Stats()

    def save_checkpoint(self, chkpt_dir, epoch):
        print('Saving..')
        state = {
            'net': self.net.state_dict(),
            'epoch': epoch,
            'stats': self.stats
        }
        if not os.path.isdir(chkpt_dir):
            os.makedirs(chkpt_dir)
        torch.save(state, os.path.join(chkpt_dir, self.chkpt_filename+'.pt'))

    def save_history(self):
        import csv
        # Save loss log
        csv_file = os.path.join(self.chkpt_dir, 'loss.csv')
        fh = open(csv_file, mode='wt')
        fields = ['epoch #', 'train', 'test']
        writer = csv.DictWriter(fh, fieldnames=fields)
        if os.path.getsize(csv_file) == 0:
            writer.writeheader()
        for i in range(len(self.stats.epochs)):
            writer.writerow({'epoch #': self.stats.epochs[i],
                'train': '{:4f}'.format(self.stats.train_epoch_loss[i]),
                'test': '{:4f}'.format(self.stats.test_epoch_loss[i]) })
        fh.close()

        # Save accuracy log
        csv_file = os.path.join(self.chkpt_dir, 'accuracy.csv')
        fh = open(csv_file, mode='wt')
        fields = ['epoch #', 'train', 'test']
        writer = csv.DictWriter(fh, fieldnames=fields)
        if os.path.getsize(csv_file) == 0:
            writer.writeheader()
        for i in range(len(self.stats.epochs)):
            writer.writerow({'epoch #': self.stats.epochs[i],
                'train': '{:4f}'.format(self.stats.train_epoch_acc[i]),
                'test': '{:4f}'.format(self.stats.test_epoch_acc[i]) })
        fh.close()

    def update(self, epoch):
        #print('Updating..')
        self.stats.epochs.append(epoch)
        self.is_best = False

        # train
        self.stats.train_epoch_loss.append(self.curr_train_loss)
        self.stats.train_epoch_acc.append(self.curr_train_acc)
        if self.curr_train_loss < self.stats.best_train_loss:
            self.stats.best_train_loss = self.curr_train_loss
            #self.is_best = True
        if self.curr_train_acc > self.stats.best_train_acc: self.stats.best_train_acc = self.curr_train_acc

        # test
        self.stats.test_epoch_loss.append(self.curr_test_loss)
        self.stats.test_epoch_acc.append(self.curr_test_acc)
        if self.curr_test_loss < self.stats.best_test_loss:
            self.stats.best_test_loss = self.curr_test_loss
            #self.is_best = True
        if self.curr_test_acc > self.stats.best_test_acc:
            self.stats.best_test_acc = self.curr_test_acc
            self.is_best = True

    def plot(self):
        print('Plotting..')
        #plt.figure()
        plt.cla()
        if len(self.stats.train_epoch_loss) > 0:
            plt.semilogy(self.stats.train_epoch_loss, label='Training')
        if len(self.stats.test_epoch_loss) > 0:
            plt.semilogy(self.stats.test_epoch_loss, label='Testing')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.chkpt_dir, 'loss.png'))

        #plt.figure()
        plt.cla()
        if len(self.stats.train_epoch_acc) > 0:
            plt.plot(self.stats.train_epoch_acc, label='Training')
        if len(self.stats.test_epoch_acc) > 0:
            plt.plot(self.stats.test_epoch_acc, label='Testing')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.chkpt_dir, 'accuracy.png'))

    #def wt_l2_norm(self):
    #    #from torch import linalg as LA
    #    for name, param in self.net.named_parameters():
    #        if param.requires_grad:
    #            print('{}: {}'.format(name, torch.norm(param.data, 'fro')))


    def load_checkpoint(self, model_path, file_name):
        # Load checkpoint.
        file_path = os.path.join(model_path, file_name)
        if not os.path.exists(file_path):
            print('file {} not found!'.format(file_path))

        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        self.net = self.net.to(self.device)

        return checkpoint, self.net

    def load_model(self):
        chkpt_dir = self.chkpt_dir
        state, self.net = self.load_checkpoint(chkpt_dir, self.chkpt_filename+'.pt')
        return self.net

    def write_to_excel(self):
        import xlsxwriter
        outdir = os.path.join(self.chkpt_dir, 'xlsx')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        filename = 'ann_nonzero_act.xlsx'
        workbook = xlsxwriter.Workbook( os.path.join(outdir, filename) )
        worksheet = workbook.add_worksheet()
        bold = workbook.add_format({'bold': True})
        row, col = 0, 0
        worksheet.write(row, col, 'layer_name', bold)
        worksheet.write(row, col+1, 'maxSpkCount', bold)
        row += 1
        for k, v in self.ann_sparsity.items():
            worksheet.write(row, col, k)
            worksheet.write(row, col+1, v)
            row += 1
        workbook.close()


    def compute_non_zeros(self, hooks, input):
        sz                       = input.size()
        b_nonzeros               = torch.count_nonzero(input, dim=(1,2,3)) # non_zeros in this batch
        nonzero_ratio_per_sample = b_nonzeros #/ torch.prod(torch.tensor(sz[1:]))
        curr_max_nonzero_ratio   = torch.max(nonzero_ratio_per_sample)
        self.ann_sparsity['input'] = max(curr_max_nonzero_ratio, self.ann_sparsity['input'])
        for i, (layer_nm, h) in enumerate(hooks):
            sz         = h.output.size()
            dims = [i for i in range(1, len(sz))]
            b_nonzeros = torch.count_nonzero(h.output, dim=dims) # non_zeros in this batch
            nonzero_ratio_per_sample   = b_nonzeros
            curr_max_nonzero_ratio     = torch.max(nonzero_ratio_per_sample)
            #print('n_nonzero_per_sample ', b_nonzeros, '\nsample size: ', sz[1:], \
            #        '\nnonzero_ratio: ', nonzero_ratio_per_sample, '\nmax_ratio: ', curr_max_nonzero_ratio)
            self.ann_sparsity[layer_nm] = max(curr_max_nonzero_ratio, self.ann_sparsity[layer_nm])


    #def test(self, get_stats=None, epoch=None, model=None):
    def test(self, epoch=None, get_stats=None, model=None):
        if model is None:
            net = self.net
        else:
            net = model
        net.eval()

        if get_stats is not None and get_stats['measure_act_sparsity']:
            keys = ['input', 'spikerelu1', 'spikerelu3', 'spikerelu5', 'spikerelu7', 'spikerelu9', \
                    'spikerelu11', 'spikerelu13', 'spikerelu16']
            relu_hooks = []
            self.ann_sparsity = {'input': 0}
            i = 0
            for n, m in net.named_modules():
                if isinstance(m, nn.ReLU):
                    relu_hooks.append( (keys[i+1], Hook(m)) )
                    self.ann_sparsity[keys[i+1]] = 0
                    i += 1

        test_loss = 0
        correct = 0
        total = 0
        max_in = 0
        confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device).float(), targets.to(self.device)
                outputs = net(inputs) # inference

                loss = self.criterion(outputs, targets)
                if self.criterion_test:
                    loss = self.criterion_test(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                epoch_loss = test_loss / (batch_idx+1)
                epoch_acc = 100*correct/total

                # confusion matrix
                for t, p in zip(targets.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                if self.interval_display:
                    if batch_idx % 100 == 0 or batch_idx == len(self.testloader)-1:
                        print('Batch idx: (%d/%d) Loss: %.3f | Acc: %.3f%% (%d/%d)' \
                                %(batch_idx, len(self.testloader), epoch_loss, epoch_acc, correct, total))
                else:
                    self.progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' \
                        % (epoch_loss, epoch_acc, correct, total))

                if get_stats is not None and get_stats['measure_act_sparsity']:
                    if batch_idx == 0: n_samples = inputs.size(0)
                    n_samples = self.compute_non_zeros(relu_hooks, inputs)


            self.curr_test_acc = epoch_acc
            self.curr_test_loss = epoch_loss

        if get_stats is not None and get_stats['measure_act_sparsity']:
            self.ann_sparsity['spikerelu16'] = 0
            self.write_to_excel()

        print('Class-wise accuracy: {}'.format(confusion_matrix.diag()/confusion_matrix.sum(1)))
        print(confusion_matrix.long())
        return epoch_acc


    def train_epoch(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        confusion_matrix = torch.zeros(self.num_classes, self.num_classes)

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            tSt = datetime.now()
            inputs, targets = inputs.to(self.device).float(), targets.to(self.device)
            #print('input shape: ', inputs.size())

            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            epoch_loss = train_loss / (batch_idx+1)
            epoch_acc = 100.*correct/total

            # confusion matrix
            for t, p in zip(targets.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            if self.interval_display:
                if batch_idx % 200 == 0 or batch_idx == len(self.trainloader)-1:
                    print('Batch idx: (%d/%d) Loss: %.3f | Acc: %.3f%% (%d/%d)' \
                            %(batch_idx, len(self.trainloader), epoch_loss, epoch_acc, correct, total))
            else:
                self.progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (epoch_loss, epoch_acc, correct, total))
            tEnd = datetime.now()
            #print('Training step time: ', (tEnd - tSt).total_seconds())

        self.curr_train_acc = epoch_acc
        self.curr_train_loss = epoch_loss
        print('Class-wise accuracy: {}'.format(confusion_matrix.diag()/confusion_matrix.sum(1)))

    def train(self):
        start_epoch = 0
        if self.resume:
            chkpt_dir = self.chkpt_dir
            state, self.net = self.load_checkpoint(chkpt_dir, self.chkpt_filename+'.pt')
            start_epoch = state['epoch'] + 1
            self.stats = state['stats']
        else:
            # Impose Kaiming He initialization
            for w in self.net.parameters():
                torch.nn.init.kaiming_uniform_(w, nonlinearity='relu')

        # verify which layers are trainable
        trainable = []
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                trainable.append(name)
        print('Trainable layers: {}'.format(trainable))

        self.net = self.net.to(self.device)
        for e in range(start_epoch, self.num_epochs):
            self.train_epoch(e)
            self.test(e)
            self.update(e)
            #self.plot()

            # Save checkpoint.
            if self.checkpoint:
                chkpt_dir = self.chkpt_dir
                if self.is_best:
                    self.save_checkpoint(chkpt_dir, e)

        self.save_history()

class Hook():
     def __init__(self, module, backward=False):
         if backward == False:
             self.hook = module.register_forward_hook(self.hook_fn)
         else:
             self.hook = module.register_backward_hook(self.hook_fn)

     def hook_fn(self, module, input, output):
         self.input = input
         self.output = output

     def close(self):
         self.hook.remove()

def ofm_sizes(net, device, input_size):
    hooks = []
    for name, layer in net.named_modules():
        #print(name)
        if isinstance(layer, nn.Conv3d):
            hooks.append((name, layer, Hook(layer)) )

    with torch.no_grad():
        in_ = torch.zeros(input_size).to(device)
        outputs = net(in_.float())

    for name, layer, h in hooks:
        print('{}:  weight- {}  stride- {}  pad- {}  output- {}'.\
                format(name, layer.weight.size(), layer.stride, layer.padding, h.output.size()))

def grad_sizes(net):
    hooks = []
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv3d):
            hooks.append((name, layer, Hook(layer, backward=True)))
    return hooks


def save_checkpoint(net, outdir, epoch, appname, stats):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
        'stats': stats
    }
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    torch.save(state, os.path.join(outdir, appname+'.pt'))


def load_checkpoint(net, model_path, file_name, device):
    #print('device: ', device)
    # Load checkpoint.
    file_path = os.path.join(model_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)

    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    net = net.to(device)
    return checkpoint, net



def print_stats(stats, train=False):
    if train:
        if stats.training.maxAccuracy is not None:
            print('Train loss: {:.4f}  accuracy: {:.3f} (best: {:.4f}, {:.3f})'.format(\
                    stats.training.loss(), stats.training.accuracy(), stats.training.minloss, stats.training.maxAccuracy))
        else:
            print('Train loss: {:.4f}  accuracy: {:.3f}'.format(stats.training.loss(), stats.training.accuracy()))
    else:
        print('Test loss: {:.4f}  accuracy: {:.3f}'.format(stats.testing.loss(), stats.testing.accuracy()))


def get_mean_and_std(dataset, numChannels=2):
     '''Compute the mean and std value of dataset.'''
     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
     mean = torch.zeros(numChannels)
     std = torch.zeros(numChannels)
     batch_num = 0
     print('==> Computing mean and std..')
     for inputs, targets in dataloader:
         for i in range(numChannels):
             mean[i] += inputs[:,i,:,:].mean()
             std[i] += inputs[:,i,:,:].std()
         print('batch num: ', batch_num)
         batch_num += 1
     mean.div_(len(dataset))
     std.div_(len(dataset))
     print('mean: {}, std: {}'.format(mean, std))
     return mean, std

def eventCountDist(dataset):
    '''Compute the mean and std value of max event count in each frame.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
    max_vals = np.zeros(len(dataset))
    #max_vals = np.zeros((len(dataset), channels))
    for i, (inputs, targets) in enumerate( dataloader ):
        b,c,h,w = inputs.size()
        max_vals[i*b : (i+1)*b] = inputs.numpy().max(axis=(1,2,3))
    print('Max event count mean: {}, std: {}, max value: {}'.format(max_vals.mean(), max_vals.std(), max_vals.max()))
    #n, bins, patches = plt.hist(x=max_vals, bins='auto')
    #plt.savefig('hist.png')


class MeanCompensator(nn.Module):
    def __init__(self, mean_std):
        super(MeanCompensator, self).__init__()
        assert len(mean_std.size()) == 2
        self.mean = mean_std[:,0]
        self.std = mean_std[:,1]

    def forward(self, x):
        B,C,H,W = x.size()
        for c in range(C):
            x[:,c,:,:] = (x[:,c,:,:] / self.std[c]) - (self.mean[c] / self.std[c])
        #print(x.size())
        #x = (x / self.std) - (self.mean / self.std)
        return x

    def extra_repr(self):
        return 'mean: {}, std: {}'.format(self.mean, self.std)


def replace_layer(net, org_layer, new_layer):
    if type(new_layer) == tuple:
        layer_num, newlayer_idx  = 0, 0
        for n,m in enumerate(net.children()):
            if isinstance(m, org_layer):
                net[layer_num] = new_layer[newlayer_idx]
                newlayer_idx += 1
            layer_num += 1
    else:
        print('need to implement this part')


def SpikeRelu(v_th, dev='cuda:0', reset='to-threshold', fire=True):
     return spikeRelu(v_th, dev, reset, fire)


class spikeRelu(nn.Module):

    #def __init__(self, v_th, dev='cuda:0', reset='to-threshold'):
    def __init__(self, v_th, dev='cuda:0', reset='to-threshold', fire=True):
        super(spikeRelu, self).__init__()
        self.threshold = v_th
        self.vmem = 0
        self.dev = dev
        self.reset = reset
        self.fire = fire

    def forward(self, x):
        # integrate the sum(w_i*x_i)
        self.vmem += x
        if not self.fire:
            return self.vmem

        # generate output spikes
        op_spikes = torch.where(self.vmem >= self.threshold, torch.ones(1, device=torch.device(self.dev)), \
                torch.zeros(1, device=torch.device(self.dev)))

        # vmem reset
        # 'reset-to-zero'
        if self.reset == 'to-zero':
            self.vmem = torch.where(self.vmem >= self.threshold, torch.zeros(1, device=torch.device(self.dev)), self.vmem)

        # 'reset-to-threshold'
        elif self.reset == 'to-threshold':
            self.vmem = torch.where(self.vmem >= self.threshold, self.vmem-self.threshold, self.vmem)

        else:
            print('Invalid reset mechanism {}'.format(reset))

        return op_spikes

    def extra_repr(self):
        return 'v_th : {}, fire : {}'.format(self.threshold, self.fire)

valid_layers = (nn.Conv2d, nn.ReLU, nn.Linear, nn.AvgPool2d, \
        nn.BatchNorm2d, nn.ReLU6, nn.AdaptiveAvgPool2d, spikeRelu, \
        Lambda)

def serialize_model(model):
    "gives relative ordering of layers in a model:"
    "layer-name => layer-type"

    name_to_type = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, valid_layers):
            name_to_type[name] = module

    return name_to_type

def compensateForMean(net, mean_std):
    new_net = copy.deepcopy(net)
    net_dict = serialize_model(new_net)
    compensate = MeanCompensator(mean_std)
    val_list = list(net_dict.values())
    val_list.insert(1, compensate)

    # take care of reshape
    for i in range(len(val_list)):
        if not isinstance(val_list[i], nn.Linear) and \
                isinstance(val_list[i+1], nn.Linear):
                    val_list.insert(i+1, Lambda(lambda x: x.view(x.size(0),-1)) )
                    break
    new_net = nn.Sequential(*val_list)
    return new_net

def visualize_data(dataset, outdir, prefix='train', fac=3200):
    from PIL import Image

    c,h,w = dataset[0][0].size()
    print(c, h, w)
    if c == 2:
        channels = 3
    else:
        channels = 1
    img = np.zeros((h, w, channels))
    print('visualize data')
    for i in range(11):
        imgpath = os.path.join(outdir, prefix+'_'+str(i)+'.png')
        #print(i)
        frame, _ = dataset[i*fac]
        #print(torch.sum(dataset[i*fac][0]))
        frame = frame[0]
        plt.savefig(imgpath)


def visualize_as_image(images, outdir='.', prefix='sample', normed=False):
    ''' `images` is a matrix of shape B,C,H,W
    with C = 1, 2 or 3 '''
    from PIL import Image

    print('image shape ', images.shape, ' event count ', np.sum(images))
    N,C,H,W = images.shape
    assert C == 3 or C == 2 or C == 1, 'given image shape is {}'.format(images.shape)
    if C == 1:
        img = np.zeros((N,H,W))
        img = images[:,0,:,:]
    elif C == 3:
        img = np.zeros((N,H,W,3))
        img[:,:,:,0] = images[:,0,:,:]
        img[:,:,:,1] = images[:,1,:,:]
        img[:,:,:,2] = images[:,2,:,:]
    else:
        img = np.zeros((N,H,W,3))
        img[:,:,:,0] = images[:,0,:,:]
        img[:,:,:,2] = images[:,1,:,:]
    if normed:
        scale = 255
    else:
        scale = 1
    img = (img*scale).astype('uint8')
    num_images = N
    for i in range(num_images):
        im = Image.fromarray(img[i])
        imgpath = os.path.join(outdir, prefix+str(i)+'.png')
        im.save(imgpath)

def create_image(filename, data, outdir):
    """ data shape: C, H, W """

    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=(0,1))
    elif len(data.shape) == 2:
        data = np.expand_dims(data, axis=(0))
    elif len(data.shape) == 3:
        pass
    else:
        print('Invalid data shape ', data.shape)
        return
    C = data.shape[0]

    plt.figure()
    grid_size = int( np.ceil( math.sqrt(C) ))
    for c in range(C):
        plt.subplot(grid_size, grid_size, c+1) #, figsize=(5,10))
        plt.imshow(data[c]) #, vmin=0, vmax=2)
    plt.savefig(os.path.join(outdir, filename+'.png'))
    plt.close()

def show_feature_maps(outdir):
    print('Creating feature map images..')
    all_files = os.listdir(outdir)
    for f in all_files:
        if f.startswith('activations_[') and f.endswith('.npz'):
            data = np.load(os.path.join(outdir, f))
            #for k, v in sorted( data.items() ):
            for k, v in data.items():
                print(k, v.shape)
                create_image(f[:-4]+k, v, outdir)

def show_spikes(outdir):
    print('Creating spike images..')
    all_files = os.listdir(outdir)
    for f in sorted( all_files ):
        #if f.startswith('activations_spike') and f.endswith('.npz'):
        if 'spike' in f and f.endswith('.npz'):
            print('\nReading spikes from file', f)
            if 'pspike' in f:
                color = 'm'
            else:
                color = 'b'
            data = np.load(os.path.join(outdir, f))
            rasters = []
            # iterate over all layers
            for k, v in data.items():
                print(k, v.shape)
                spatial_v = np.sum(v, axis=(-1))
                create_image(f[:-4]+k, spatial_v, outdir)
                plot_spike_raster(v, os.path.join(outdir, 'raster_'+f[12:-4]+k), color)
                break


def plot_spike_raster(spike_arr, name, color='b'):
    print('total spikes ', np.sum(spike_arr))
    if len(spike_arr.shape) == 1:
        spike_arr = np.expand_dims(spike_arr, axis=(0,1,2))
    elif len(spike_arr.shape) == 2:
        spike_arr = np.expand_dims(spike_arr, axis=(0,1))
    elif len(spike_arr.shape) == 3:
        spike_arr = np.expand_dims(spike_arr, axis=(0))
    elif len(spike_arr.shape) == 4:
        pass
    else:
        print('Invalid data shape ', data.shape)
        return

    C, H, W, T = spike_arr.shape
    N = H * W
    for c in range(C):
        plt.figure()
        plt.autoscale(False, axis='y')
        mat = spike_arr[c].reshape(-1, T) # N, T
        #print('mat size ', mat.shape)
        for n in range(N):
            plt.plot((n+1)*mat[n,:] , '|'+color, markersize=3)
        plt.ylim(bottom=0.6, top=N)
        plt.savefig(name+'_ch{}.png'.format(c))
        plt.close()

#def plot_spike_raster(spike_arr, name):
#    ''' spike_arr should be 2D of shape
#    N, T where N = num neurons, T = spikes'''
#
#    #print('plotting raster..')
#    #assert len(spike_arr.shape) == 2
#    #plt.figure(figsize=(5, 8))
#    plt.figure()
#    C = len(spike_arr)
#    for c in range(C):
#        N, T = spike_arr[c].shape
#        plt.subplot(C, 1, c+1)
#        for n in range(N):
#            plt.plot((n+1)*spike_arr[c][n,:] , '.b', markersize=1)
#    plt.ylim(bottom=0.6)
#    plt.savefig(name+'.png')
#    plt.close()

def MSE(tensor1, tensor2):
    diff = tensor1 - tensor2
    #print(diff[0,0,0:10,0:10,0])
    diff_2 = diff * diff
    print(diff[0,0,0:10,0:10,0])
    #print(diff_2.size())
    print('sum of squares of difference: ', torch.sum(diff_2))
    rms = torch.sqrt(diff_2.mean())
    return rms

class SNNConverter:
    def __init__(self, **kwargs):
        self.net = kwargs['net']
        self.device = kwargs['device']
        self.dataloader = kwargs['dataloader'] # trainloader
        self.outdir = kwargs.get('outdir', '.')
        self.criterion = kwargs.get('criterion', None)
        self.len_dataset = kwargs.get('len_dataset', 10000)
        self.percentile = kwargs.get('percentile', 99.9)
        self.simulation_params = kwargs.get('simulation_params', None)
        self.valLoader = kwargs.get('valLoader', None)
        self.batchsize = kwargs.get('batchsize', 32)
        self.num_classes = kwargs.get('num_classes', 11)
        self.aedatTestLoader = kwargs.get('aedatTestLoader', None)
        self.aedatTestLoaderRaw = kwargs.get('aedatTestLoaderRaw', None)
        self.mean_std = kwargs.get('mean_std', None)
        self.resetTime = kwargs.get('resetTime', None)
        self.datatype = kwargs.get('datatype', None)
        self.fire = kwargs.get('fire_last_layer', False)
        self.class_to_eventCnt_dict = kwargs.get('classToEventCount', None)

        self.layers = (nn.Conv2d, nn.AvgPool2d, nn.Linear)
        self.net.eval() # NOTE: important line to ensure weights DO NOT change during inference
        self.spike_vector = None

    def compute_thresholds(self, model=None, percentile=99.9):
        print('\n[INFO] Computing SNN thresholds from ANN activations..')
        if model is None:
            net = self.net
        else:
            net = model
        relus = []
        start = False
        if self.mean_std is not None:
            for name, module in net.named_modules():
                if isinstance(module, MeanCompensator):
                    relus.append(module)
                    start = True
                if start and isinstance(module, self.layers):
                    relus.append(module)
        else:
            name_to_type = serialize_model(net)
            key_list = list(name_to_type.keys())

            for i in range(len(key_list)):
                if i < len(key_list)-1:
                    next_layer = name_to_type[key_list[i+1]]
                    if isinstance(next_layer, nn.ReLU):
                        relus.append(name_to_type[key_list[i]])
                else:
                    relus.append(name_to_type[key_list[i]])

        hooks = [Hook(layer) for layer in relus]
        print('number of spike layers with thresholds: {}'.format(len(hooks)))

        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        acts = np.zeros((len(hooks)+1, self.len_dataset))
        with torch.no_grad():
            for n, (inputs, targets) in enumerate(self.dataloader):
                inputs, targets = inputs.to(self.device).float(), targets.to(self.device)
                outputs = net(inputs)
                #print('inputs ', inputs.size(), ' targets ', targets.size(), ' outputs ', outputs.size() )
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                epoch_loss = test_loss / (n+1)
                epoch_acc = 100*correct/total

                batch_size = targets.size(0)
                if len(inputs.size()) == 4:
                    img_max = np.amax(inputs.cpu().numpy(), axis=(1,2,3))
                else:
                    img_max = np.amax(inputs.cpu().numpy(), axis=(1,2))
                acts[0,n*batch_size:(n+1)*batch_size] = img_max

                for i, hook in enumerate(hooks):
                    #print('hook output ', hook.output.size(), len(hook.output.size()))
                    if len(hook.output.size()) == 3:
                        acts[i+1][n*batch_size:(n+1)*batch_size] = np.amax(hook.output.cpu().numpy(), axis=(1,2))
                    elif len(hook.output.size()) == 4:
                        acts[i+1][n*batch_size:(n+1)*batch_size] = np.amax(hook.output.cpu().numpy(), axis=(1,2,3))
                    else:
                        acts[i+1][n*batch_size:(n+1)*batch_size] = np.amax(hook.output.cpu().numpy(), axis=1)

                batch_idx = n
                if batch_idx % 100 == 0 or batch_idx == len(self.dataloader)-1:
                    print('Batch idx: (%d/%d) Loss: %.3f | Acc: %.3f%% (%d/%d)' \
                            %(batch_idx, len(self.dataloader), epoch_loss, epoch_acc, correct, total))

        max_val = np.percentile(acts, self.percentile, axis=1)
        print('{}th percentile of max activations: {}'.format(self.percentile, max_val))
        filenm = 'max_acts_{}.txt'.format(self.percentile)
        np.savetxt(os.path.join(self.outdir, filenm), max_val, fmt='%.5f')

        thresholds = torch.zeros(len(max_val)-1)
        for i in range(len(thresholds)):
            thresholds[i] = max_val[i+1] / max_val[i]
        np.savetxt(os.path.join(self.outdir, 'thresholds_{}.txt'.format(self.percentile)), thresholds, fmt='%.5f')
        print('thresholds: ', thresholds)



    def adjust_weights(self, wt_layer, bn_layer):
        num_out_channels = wt_layer.weight.size()[0]

        bias = torch.zeros(num_out_channels)
        wt_layer_bias = torch.zeros(num_out_channels)
        if wt_layer.bias is not None:
            wt_layer_bias = wt_layer.bias

        wt_cap = torch.zeros(wt_layer.weight.size())
        for i in range(num_out_channels):
            beta, gamma = 0, 1

            if bn_layer.weight is not None:
                gamma = bn_layer.weight[i]
            if bn_layer.bias is not None:
                beta = bn_layer.bias[i]

            sigma = bn_layer.running_var[i]
            mu = bn_layer.running_mean[i]
            eps = bn_layer.eps
            scale_fac = gamma / torch.sqrt(eps+sigma)
            wt_cap[i,:,:,:] = wt_layer.weight[i,:,:,:]*scale_fac
            bias[i] = (wt_layer_bias[i]-mu)*scale_fac + beta
        return (wt_cap, bias)


    def merge_bn(self, model_nobn):

        "merges bn params with those of the previous layer"
        "works for the layer pattern: conv->bn only"

        model = self.net

        # Serialize the original model
        name_to_type = serialize_model(model)
        key_list = list(name_to_type.keys())

        # Serialize the nobn model
        name_to_type_nobn = serialize_model(model_nobn)
        conv_names = []
        for k,v in name_to_type_nobn.items():
            if type(v) == nn.Conv2d or type(v) == nn.Linear:
                conv_names.append(k)

        nobn_num = 0
        layer_num = 0
        for i,n in enumerate(key_list):
            if isinstance(name_to_type[n], nn.Conv2d) and \
                    isinstance(name_to_type[key_list[i+1]], nn.BatchNorm2d):

                conv_layer = name_to_type[n]
                bn_layer = name_to_type[key_list[i+1]]
                new_wts, new_bias = self.adjust_weights(conv_layer, bn_layer)

                nobn_name = conv_names[layer_num]
                conv_layer_nobn = name_to_type_nobn[nobn_name]

                conv_layer_nobn.weight.data = new_wts
                if conv_layer_nobn.bias is not None:
                    conv_layer_nobn.bias.data = new_bias

                layer_num += 1

            elif isinstance(name_to_type[n], nn.Conv2d) or \
                    isinstance(name_to_type[n], nn.Linear):
                layer = name_to_type[n]

                nobn_name = conv_names[layer_num]

                layer_nobn = name_to_type_nobn[nobn_name]
                layer_nobn.weight.data = layer.weight.data.clone()
                if layer.bias is not None and layer_nobn.bias is not None:
                    layer_nobn.bias.data = layer.bias.data.clone()

                layer_num += 1

        return model_nobn

    def convert_to_snn(self):
        # insert relu layers at appropriate places if they dont exist
        spikenet = copy.deepcopy(self.net)
        net_dict = serialize_model(spikenet)
        key_list = list(net_dict.keys())
        val_list = list(net_dict.values())

        # MUST have the corresponding Relus.

        # simply appending a relu after the last linear layer
        if isinstance(val_list[-1], nn.Linear):
            val_list.append(nn.ReLU())

        # take care of reshape
        for i in range(len(val_list)):
            if not isinstance(val_list[i], nn.Linear) and \
                    isinstance(val_list[i+1], nn.Linear):
                        val_list.insert(i+1, Lambda(lambda x: x.view(x.size(0),-1)) )
                        break

        spikenet = nn.Sequential(*val_list)
        num_relus = 0
        for c in spikenet.children():
            if isinstance(c, nn.ReLU):
                num_relus += 1

        # read the threshold file
        th_filenm = 'thresholds_{}.txt'.format(self.percentile)
        th_filepath = os.path.join(self.outdir, th_filenm)
        if not os.path.exists(th_filepath):
            print('WARNING: thresholds file {} does not exist. Setting thresholds to 0'.format(th_filenm))
            thresholds = np.zeros((num_relus))
        else:
            thresholds = np.loadtxt(th_filepath)

        # create instances of SpikeReLUs with the above thresholds
        spike_relus = []
        for i in range(num_relus):
            if i == num_relus-1 and not self.fire:
                # to indicate whether or not the last layer will fire
                spike_relus.append(SpikeRelu(thresholds[i], dev=self.device, fire=False))
            else:
                spike_relus.append(SpikeRelu(thresholds[i], dev=self.device, fire=True))
        spike_relus = tuple(spike_relus)
        #print(spike_relus)

        # replace all relu layers with spikeRelu
        replace_layer(spikenet, org_layer=nn.ReLU, new_layer=spike_relus)

        ## NOTE: adding the compensator insertion part here
        if self.mean_std is not None:
            spikenet = compensateForMean(spikenet, self.mean_std)

        # adjust the weights and biases
        snn_map = {}
        i = 0
        for m in spikenet.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.AvgPool2d)):
                snn_map[i] = m
                i += 1
        ann_map = {}
        i = 0
        for m in self.net.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.AvgPool2d)):
                ann_map[i] = m
                i += 1

        maxact_file = 'max_acts_{}.txt'.format(self.percentile)
        maxact_filepath = os.path.join(self.outdir, maxact_file)
        max_acts = np.loadtxt(maxact_filepath)
        for num, (k,v) in enumerate(snn_map.items()):
            if isinstance(v, (nn.Conv2d, nn.Linear)):
                v.weight.data = ann_map[k].weight.data.clone()
                if v.bias is not None:
                    temp = ann_map[k].bias.data.clone()
                    v.bias.data = nn.Parameter( temp / max_acts[num] )
        return spikenet

    def poisson_spikes(self, images, MFR=1, device='cuda:0'):
        ''' creates poisson spikes from input images '''
        ''' shape of input and output: BCHWT '''
        #images = images.cpu()
        random_vals = torch.rand(images.size(), device=device)
        images_max = images.cpu().numpy().max(axis=(1,2,3), keepdims=True)
        images_max = torch.from_numpy(images_max).to(device)
        ratio = (abs(images) / images_max) * MFR
        #print(ratio.size())
        random_vals = torch.where(random_vals <= ratio, torch.tensor(1, device=device), \
                torch.tensor(0, device=device))
        return random_vals


    def create_mats(self, net, img_size, hooks):
        inp_size = img_size[:-1]

        inp = torch.zeros(inp_size).to(self.device)
        outputs = net(inp.float())

        mats = []
        T = img_size[-1]
        num_neurons = np.prod(inp_size)
        mats.append(np.zeros((num_neurons, T)))
        for h in hooks:
            shape = h.output.size()
            num_neurons = np.prod(shape)
            temp = np.zeros((num_neurons, T))
            mats.append(temp)
        return mats


    def record_spikes_fn(self, spikenet, reset_time, input_sz):
        ''' input_sz is of the form 1,CHWT '''

        print('=== Creating spike buffers for input with size: {} ==='.format(input_sz))
        relus = []
        for m in spikenet.modules():
            if isinstance(m, spikeRelu):
                relus.append(m)

        hooks = [Hook(layer) for layer in relus]
        mats = self.create_mats(spikenet, input_sz, hooks)
        return hooks, mats


    def simulate_with_poisson_spikes(self, spikenet, time_window):
        total_images, total_correct, expected_correct = 0, 0, 0
        confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        print('== Simulation time window {} =='.format(time_window))
        # simulate the model
        with torch.no_grad():
            for batch_idx, (images, target) in enumerate(self.valLoader):
                print ('\n\n------------ inferring batch {} -------------'.format(batch_idx))

                # perform inference on the original model to compare
                images, target = images.to(self.device), target.to(self.device)
                output_org = self.net(images.float())
                _, predicted_org = output_org.max(1)
                #print('images ', images.size())

                # create the spiking model
                spikenet = (self.convert_to_snn()).to(self.device)
                spikenet.eval()
                #print(spikenet)

                batch_size = images.size(0)
                out_spikes_t_b_c = torch.zeros((time_window, batch_size, self.num_classes), device=self.device)
                total_spikes_b_c = torch.zeros((batch_size, self.num_classes), device=self.device)
                rpt = images.unsqueeze(-1)
                p_device = self.device
                if self.simulation_params['optimize_mem'] >= 1:
                    rpt = rpt.cpu()
                    p_device = 'cpu'

                repeats_along_dims = []
                for i in range(len(rpt.size())-1):
                    repeats_along_dims.append(1)
                repeats_along_dims.append(time_window)
                #print('repeats along dims ', repeats_along_dims)
                image_repeat = rpt.repeat(*repeats_along_dims)
                #image_repeat = rpt.repeat((1,1,1,1,time_window))
                #print(image_repeat.size())

                p_spikes = self.poisson_spikes(image_repeat, device=p_device)
                if self.simulation_params['optimize_mem'] == 1:
                    p_spikes = p_spikes.to(self.device)

                ########## SNN inference (starts) ##########

                # first few layers have 'delayed' spikes
                if self.delayed_spikes:
                    ifm_spike = p_spikes.float() # B,C,H,W,T
                    delayed_layers, spikenet_new = self.bifurcate_model(spikenet, \
                            self.num_delayed_layers)
                    #print('delayed layers ', delayed_layers)
                    #print(spikenet_new)

                    # SNN inference on 'delayed' spikes (starts)
                    for i in range(self.num_delayed_layers):
                        delayed_layer = delayed_layers[i]
                        next_layer = delayed_layers[i+1]
                        vth = next_layer.threshold
                        if isinstance(next_layer, spikeRelu):
                            # initializing vmem with zeros
                            vmem = delayed_layer(torch.zeros(p_spikes.size()[:-1], device=self.device))
                            ofm_spike = torch.zeros((*vmem.size(), time_window), device=self.device)

                            # accumulate vmem, dont fire
                            for t in range(time_window):
                                #spikes = ifm_spike[:,:,:,:,t]
                                spikes = ifm_spike[...,t]
                                #print('vmem: ', vmem[0,0,0:10,0])
                                # vmem updates every time step, thresholding delayed
                                vmem += delayed_layer(spikes) # delayed layer forward pass B,C,H,W

                                ## NOTE: just checking
                                #ofm_spike[:,:,:,:,t] = torch.where(vmem >= vth, torch.tensor(1, \
                                #        device=self.device), torch.tensor(0, device=self.device))
                                #vmem = torch.where(vmem >= vth, vmem - vth, vmem)

                            # threshold accumulated vmem and fire
                            for t in range(time_window):
                                #ofm_spike[:,:,:,:,t] = torch.where(vmem >= vth, torch.tensor(1, \
                                ofm_spike[...,t] = torch.where(vmem >= vth, torch.tensor(1, \
                                        device=self.device), torch.tensor(0, device=self.device))
                                vmem = torch.where(vmem >= vth, vmem - vth, vmem)

                            ifm_spike = ofm_spike
                    # SNN inference on 'delayed' spikes (ends)

                    # simulate the rest of the SNN in the usual way (starts)
                    for t in range(time_window):
                        outspikes = spikenet_new(ofm_spike[...,t])
                        if self.fire:
                            total_spikes_b_c += outspikes
                        else:
                            total_spikes_b_c = outspikes
                    # simulate the rest of the SNN in the usual way (ends)

                else:
                    # starting of time-stepped spike integration of SNN
                    for t in range(time_window):

                        # convert image pixels to spiking inputs
                        #spikes = p_spikes[:,:,:,:,t]
                        spikes = p_spikes[...,t]
                        # NOTE: uncomment the following line if CUDA runs out of memory while allocating p_spikes
                        #if self.simulation_params['optimize_mem'] == 2:
                        #    spikes = p_spikes[:,:,:,:,t].to(self.device)
                        #print(spikes)

                        # TODO: provide a switch here
                        out_spikes = spikenet(spikes.float()) # using poisson input
                        #out_spikes = spikenet(images.float()) # using analog input
                        out_spikes_t_b_c[t,:,:] = out_spikes
                        #print(out_spikes)
                        if self.fire:
                            total_spikes_b_c += out_spikes
                        else:
                            total_spikes_b_c = out_spikes
                    # end of time-stepped spike integration

                #return
                ########## SNN inference (ends) ##########

                _, predicted = torch.max(total_spikes_b_c.data, -1)

                # confusion matrix
                for t, p in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                print('snn prediction: ', predicted.cpu().numpy())
                print('ann prediction: ', predicted_org.cpu().numpy())
                print('ground truth  : ', target.cpu().numpy())
                total_images += target.size(0)
                total_correct += predicted.eq(target).sum().item()
                expected_correct += predicted_org.eq(target).sum().item()

                model_accuracy = total_correct / total_images * 100
                expected_acc = expected_correct / total_images * 100
                print('Model accuracy on {} test images: {:.3f}%\t\t\tacc. of ANN: {:.3f}%'.format(total_images, model_accuracy, expected_acc))
                #break

            #print('ann total correct: ', expected_correct)
            print('Overall accuracy on {} test images: {:.3f}%\t\t\tacc. of ANN: {:.3f}%'.format(total_images, model_accuracy, expected_acc))
            print('Class-wise accuracy: {}'.format(confusion_matrix.diag()/confusion_matrix.sum(1)))
            print(confusion_matrix.long())


    def create_spike_image(self, input, outdir, index):
        c,h,w,t = input.size()
        image = torch.sum(input, dim=(-1))
        image /= torch.max(image)
        image = image.cpu().numpy()

        img = np.zeros((h, w, 3))
        img[:,:,0] = image[0,:,:]
        if c == 2:
            img[:,:,2] = image[1,:,:]
        img_arr = (img * 255).astype('uint8')
        im = Image.fromarray(img_arr) # , mode='L')
        outdir1 = os.path.join(self.outdir, outdir)
        if not os.path.exists(outdir1):
            os.makedirs(outdir1)
        filepath = os.path.join(outdir1, str(index) + '.png')
        im.save(filepath)


    def create_mats(self, net, img_size, relus, spike=False):

        #print('img_size ', img_size, ' spike ', spike)
        hooks = [Hook(layer[1]) for layer in relus]
        if spike:
            T = img_size[-1]
            img_size = list(img_size[:-1])
        inp = torch.zeros(img_size).to(self.device)
        outputs = net(inp.float())
        inp = None

        if spike:
            img_size.append(T)
        mats = {}
        mats['input'] = torch.zeros(img_size[1:])
        for i, (n, m) in enumerate(relus):
            shape = hooks[i].output.size()
            curr_shape = list(shape[1:])
            if spike:
                curr_shape.append(T)
            #print(n, 'curr_shape ', curr_shape)
            temp = torch.zeros(curr_shape)
            mats[n] = temp

        return hooks, mats

    def create_buffers(self, net, img_size, spike=False):
        """ creates buffers to store intermediate activation values """
        """ img_size: 1,C,H,W  if spike = True, else 1,C,H,W,T      """
        """ if spike = True --> `net` is SNN else ANN               """
        """ returns hooks: dict(module_name --> layer)              """
        """ mats: dict(module_name --> matrix of feature map size)  """

        relus = []
        if spike:
            i = 0
            for n, m in net.named_modules():
                if isinstance(m, spikeRelu):
                    relus.append(('spikerelu'+n, m))
        else:
            classifiers = []
            for n, m in net.named_modules():
                if isinstance(m, nn.Linear):
                    classifiers.append((n, m))

            for n, m in net.named_modules():
                if isinstance(m, nn.ReLU):
                    relus.append((n, m))
            relus.append(classifiers[-1])

        hooks, mats = self.create_mats(net, img_size, relus, spike)
        return hooks, mats

    def save_activations(self, net, hooks, buffers, image, fileprefix):
        """ image: tensor of shape 1,C,H,W     """
        assert image.shape[0] == 1, 'batch size should be 1. Given image shape {}'.format(image.shape)

        for i, (name, _) in enumerate(buffers.items()):
            if i == 0:
                buffers[name] = image.cpu().numpy()[0]
            else:
                layer_out = hooks[i-1].output[0]
                buffers[name] = layer_out.cpu().numpy()
        np.savez(os.path.join(self.outdir, fileprefix + '.npz'), **buffers)

    def save_spikes(self, spikenet, shooks, sbuffers, dvs_spikes, time):
        """ dvs_spikes: tensor of shape 1,C,H,W     """
        assert dvs_spikes.shape[0] == 1, 'batch size should be 1. Given image shape {}'.format(dvs_spikes.shape)

        for i, (name, _) in enumerate(sbuffers.items()):
            if time < sbuffers[name].size()[-1]:
                if i == 0:
                    layer_out = dvs_spikes[0]
                else:
                    layer_out = shooks[i-1].output[0]
                sbuffers[name][...,time] = layer_out
        return sbuffers

    def measure_sparsity(self, hooks, buffers, image, batch_size, sparsity_dict, prev_data_size, spike=False):
        """ image: tensor of shape 1,C,H,W (T)   """

        print('Measure sparsity')
        new_data_size = prev_data_size + batch_size
        for i, (name, v) in enumerate(buffers.items()):
            if i == 0:
                tensor = image
            else:
                if spike:
                    tensor = v
                else:
                    tensor = hooks[i-1].output[0]
            non_zeros      = torch.nonzero(tensor).size(0)
            ifm_size       = torch.prod(torch.tensor(tensor.size()))
            zeros          = ifm_size - non_zeros
            curr_sparsity  = zeros.float() / ifm_size.float()
            prev_sparsity  = sparsity_dict[name]
            new_sparsity   = (prev_sparsity * prev_data_size + curr_sparsity) / new_data_size
            sparsity_dict[name] = new_sparsity
            print('layer: {}, non_zeros: {}'.format(name, non_zeros))
        prev_data_size = new_data_size
        return sparsity_dict, prev_data_size

    def save_ops(self, hooks, buffers, image, batch_size, label):
        label = str(label)
        for i, (name, v) in enumerate(buffers.items()):
            if i == len(buffers.keys())-1: break
            if i == 0:
                tensor = image
                num_samples = self.spike_rate[name][label][1] + batch_size
            else:
                tensor = v
            num_spikes = torch.sum(tensor).cpu().item()
            num_neurons = torch.prod(torch.tensor(tensor.size()[:-1]))
            spike_rate = num_spikes / num_neurons
            cum_spike_rate = self.spike_rate[name][label][0] + spike_rate
            self.spike_rate[name][label] = (cum_spike_rate, num_samples)
            #print('layer: {}, num_spikes: {}, num_neurons: {}, spikerate: {:.4f}'.format(name, \
            #    num_spikes, num_neurons, spike_rate))
            reduction_dims = [i for i in range(len(tensor.size()[:-1]))]
            spikes_per_time = torch.sum(tensor, dim=reduction_dims)
            #print(spikes_per_time)
            self.max_spike_count[name] = max(torch.max(spikes_per_time), self.max_spike_count[name])
            #print('layer: {}, tensor_shape: {}, spikes_per_time shape: {}, max_spike_count: {}'.format(name, tensor.size(), \
            #        spikes_per_time.size(), torch.max(spikes_per_time)))


    def measure_ops(self, buffers):
        keys = list(self.spike_rate.keys())
        for i, (layer_name, v) in enumerate( self.spike_rate.items() ):
            if i < len(keys)-1:
                next_layer_size = torch.prod(torch.tensor(buffers[keys[i+1]].size()))
                #print(v)
                for label, vals in v.items():
                    #print(label, vals)
                    if vals[1] == 0:
                        overall_spike_rate = 0
                    else:
                        overall_spike_rate = vals[0].item() / vals[1]
                    self.spike_rate[layer_name][label] = (overall_spike_rate, vals[1])

    def writeOpsToExcel(self, sbuffers, delayed_spikes):
        import xlsxwriter
        if delayed_spikes:
            if self.delayed_cyclic: prefix = 'del_cyc_'
            elif self.delayed_piped: prefix = 'del_G{}_K{}_type3A_'.format(self.gate_time, \
                    self.simulation_params['fire_time'])
            # NOTE: only simulating type3A for now
            #elif self.delayed_piped: prefix = 'del_G{}_K{}'.format(self.gate_time, \
            #        self.simulation_params['fire_time'])
        elif self.simulation_params['combine_frames']:
            prefix = 'combine_'+ str(self.simulation_params['combine_factor']) + '_'
        else: prefix = 'plain_'
        outdir = os.path.join(self.outdir, 'xlsx')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        filename = prefix + 'spikerate.xlsx'
        workbook = xlsxwriter.Workbook( os.path.join(outdir, filename) )
        worksheet = workbook.add_worksheet()
        # Add a bold format to use to highlight cells.
        bold = workbook.add_format({'bold': True})
        print('\n------- Saving spike rates to xlsx file {} -------'.format(filename))

        col_heads = ['layer_name']
        for i in range(1,12):
            col_heads.append(str(i))
        col_heads += ['wt_avg', 'ANN_Ops', 'SNN_Ops', 'SNN/ANN_Ops', 'maxSpkCount']

        # write column headings
        row = 0
        for col, item in enumerate(col_heads):
            worksheet.write(row, col, item, bold)
            if col == 0: print('{:11}'.format(item), end=': ')
            else: print('{:8}'.format(item), end=': ')

        self.measure_ops(sbuffers)

        pre_print = ''
        ann = np.loadtxt('ANN_Ops.txt', dtype=str)
        tot_ann_ops, tot_snn_ops = 0, 0
        for layer_name, v in self.spike_rate.items():
            # row heading
            row += 1
            col = 0
            worksheet.write(row, col, layer_name, bold)

            if col == 0: pre_print = '\n'
            else: pre_print = ''
            print('{}{:11}'.format(pre_print, layer_name), end=': ')

            wt_avg, tot_samples = 0, 0
            for label, vals in v.items():
                col += 1
                spk_rate, num_samples = vals
                #print(spk_rate, num_samples)
                #print()
                wt_avg += spk_rate*num_samples
                tot_samples += num_samples
                worksheet.write(row, col, vals[0])
                print('{:8.4f}'.format(vals[0]), end=', ')

            #print(row)
            if row <= 8:
                col += 1
                wt_avg /= tot_samples
                worksheet.write(row, col, wt_avg)
                print('{:8.4f}'.format(wt_avg), end=', ')
                col += 1
                ann_ops = ann[row-1, 1].astype('float')
                tot_ann_ops += ann_ops
                worksheet.write(row, col, ann_ops)
                print('{:3.2e}'.format(ann_ops), end=', ')
                col += 1
                snn_ops = ann_ops * wt_avg
                tot_snn_ops += snn_ops
                worksheet.write(row, col, snn_ops)
                print('{:3.2e}'.format(snn_ops), end=', ')
                col += 1
                snn_ann_ratio = snn_ops / ann_ops
                worksheet.write(row, col, snn_ann_ratio)
                print('{:8.4f}'.format(snn_ann_ratio), end=', ')
                col += 1
                worksheet.write(row, col, self.max_spike_count[layer_name])
                print('{:8}'.format(self.max_spike_count[layer_name]), end=', ')
                #print('{:8.4f}'.format(self.max_spike_count[layer_name]), end=', ')

        row += 1
        for col in range(0, 12):
            if col == 0:
                item = '{:11}'.format('numSamples')
                pre_print = '\n'
                worksheet.write(row, col, item, bold)
            else:
                item = self.spike_rate['input'][str(col-1)][1]
                pre_print = ''
                worksheet.write(row, col, item)
            print('{}{:8}'.format(pre_print, item), end=', ')

        if row > 8:
            col += 1
            print('{:8}'.format('sumOps'), end=', ')
            worksheet.write(row, col, 'sumOps', bold)
            col += 1
            worksheet.write(row, col, tot_ann_ops)
            print('{:3.2e}'.format(tot_ann_ops), end=', ')
            col += 1
            worksheet.write(row, col, tot_snn_ops)
            print('{:3.2e}'.format(tot_snn_ops), end=', ')
            col += 1
            tot_snn_ann_ratio = tot_snn_ops / tot_ann_ops
            worksheet.write(row, col, tot_snn_ann_ratio)
            print('{:8.4f}'.format(tot_snn_ann_ratio), end=', ')

        print('\n')
        workbook.close()


    def sort_dict(self, dict_):
        keys  = []
        for k,v in dict_.items():
            if k != 'input':
                keys.append(k)
        def layerNum(elem):
            return int(elem[9:])
        keys = sorted(keys, key=layerNum)
        new_dict = {'input': dict_['input']}
        for k in keys:
            new_dict[k] = dict_[k]
        return new_dict


    def bifurcate_model(self, spikenet, num_delayed_layers):
        """ breaks model into a list(delayed spike layers) + remaining SNN model """
        net_dict = serialize_model(spikenet)

        delayed_layers = OrderedDict() # changing list to dict.
        key_list = list(net_dict.keys())
        for i in range(2 * num_delayed_layers): # every conv/avg layer is followed by relu
            delayed_layer = copy.deepcopy(net_dict[key_list[i]])
            delayed_layers[key_list[i]] = delayed_layer
            net_dict.pop(key_list[i])
        spikenet_new = nn.Sequential(net_dict)
        delayed_model = nn.Sequential(delayed_layers)

        return delayed_model, spikenet_new

    def simulate_delayed_spikes_cyclic(self, spikenet, in_spikes, num_delayed_layers, batch_idx=0, \
            target=-1):
        ''' in_spikes: tensor of shape 1,C,H,W,T             '''
        ''' spikenet: SNN                                    '''
        ''' num_delayed_layers: # layers with delayed spikes '''

        _, C, H, W, T = in_spikes.size()
        G = total_time = T

        delayed_model, spikenet_new = self.bifurcate_model(spikenet, num_delayed_layers)
        delayed_model_dict = serialize_model(delayed_model)
        delayed_model_keys = list(delayed_model_dict)

        in_spikes_tchw = in_spikes.permute(0,4,1,2,3).squeeze(0) # T,C,H,W
        ifm = in_spikes_tchw # T,C,H,W

        # looping over all frame repeats (start)
        for rf in range (self.RF):
            ifm = in_spikes_tchw # T,C,H,W
            ########## process delayed layers (start)
            i = 0
            for del_name, layer in delayed_model.named_children():
                if isinstance(layer, (nn.Conv2d, nn.AvgPool2d, nn.Linear)):
                    if rf == 0:
                        ######################## (inference on delayed layer)
                        ofm = layer(ifm) # T,C,H,W

                        if self.simulation_params['measure_sparsity'] or self.simulation_params['record_spikes'] \
                            or self.simulation_params['measure_ops']:
                            dhooks, dbuffers = self.create_buffers(delayed_model, img_size=(1,C,H,W,T), spike=True)
                            shooks, sbuffers = self.create_buffers(spikenet_new, img_size=(1,*ofm.size()[1:],T), \
                                    spike=True)

                        ######################## (vmem after delayed inference)
                        # we want to accu. the vmem only for a fixed time window and not the entire
                        # input duration.
                        pre_act_parts = torch.split(ofm, G, dim=0) # tuple of len ceil(T/G)
                        vprev = torch.zeros((1, *ofm.size()[1:]), device=self.device)

                    ###### dissipate vmem (start)
                    ofm_spike = torch.zeros((1, *ofm.size()[1:], total_time), device=self.device) # 1,C,H,W,T
                    vth = delayed_model_dict[delayed_model_keys[i+1]].threshold
                    #print('delayed_layer vth: ', vth)
                    for P, pre_act in enumerate(pre_act_parts):
                        vmem = vprev + torch.sum(pre_act, axis=0) # membrane potential 1,C,H,W
                        Gcurr = G
                        if P == len(pre_act_parts)-1:
                            Gcurr = T - P*G

                        for tau in range(Gcurr):
                            t = P*G + tau
                            ofm_spike[0,:,:,:,t] = torch.where(vmem >= vth, torch.tensor(1, device=self.device), \
                                    torch.tensor(0, device=self.device))
                            vmem = torch.where(vmem >= vth, vmem - vth, vmem)

                            if self.simulation_params['record_spikes'] or \
                                    (self.simulation_params['measure_sparsity'] and rf == 0) \
                                    or self.simulation_params['measure_ops']:
                                self.save_spikes(layer, dhooks, dbuffers, in_spikes[...,t], t)

                        vprev = vmem # residual potential from previous gated window
                    ###### dissipate vmem (end)

                    ifm = ofm_spike.permute(0,4,1,2,3).squeeze(0) # T,C,H,W
                i = i + 1
            ########## process delayed layers (end)

            total_event_spikes = torch.zeros((1, self.num_classes), device=self.device)
            spike_sum = torch.sum(ofm_spike, dim=-1)
            #print(spike_sum.size())
            rem_dur = max_spike_count = torch.max(spike_sum)
            self.max_rem_dur = max(rem_dur, self.max_rem_dur)
            print('remaining duration: {}, max remaining duration: {}'.format(rem_dur, self.max_rem_dur))
            # simulate the remaining layers with original spikes
            for t in range(int( rem_dur )):
                outspikes = spikenet_new(ofm_spike[...,t])
                if self.fire:
                    total_event_spikes += outspikes
                else:
                    total_event_spikes = outspikes

                if self.simulation_params['record_spikes'] or \
                        (self.simulation_params['measure_sparsity'] and rf == 0) \
                        or self.simulation_params['measure_ops']:
                    sbuffers = self.save_spikes(spikenet_new, shooks, sbuffers, ofm_spike[...,t], t)
            # simulate the remaining layers with original spikes (end)

            # unifying both buffers
            # save to file the DVS spikes created during SNN simulation
            if self.simulation_params['record_spikes'] or \
                    (self.simulation_params['measure_sparsity'] and rf == 0) \
                    or self.simulation_params['measure_ops']:
                sbuffers['spikerelu1'] = sbuffers.pop('input')
                for k, v in dbuffers.items():
                    if k == 'input':
                        sbuffers[k] = dbuffers[k]
                sbuffers = self.sort_dict(sbuffers)

            # measure sparsity of all layers
            if self.simulation_params['measure_sparsity'] and rf == 0:
                self.snn_sparsity, self.snn_prev_data_size = self.measure_sparsity(shooks, sbuffers, \
                        in_spikes, 1, self.snn_sparsity, self.snn_prev_data_size, spike=True)

            # save OPs measured for current batch
            if self.simulation_params['measure_ops']:
                self.save_ops(shooks, sbuffers, in_spikes, 1, target.cpu().item())

            if self.simulation_params['record_spikes']:
                for k, v in sbuffers.items():
                    print(k, v.shape, torch.sum(v).item())
                    sbuffers[k] = sbuffers[k].cpu().numpy()

                G = self.simulation_params['gate_time']
                if G < 0:
                    type_ = ''
                else:
                    type_ = 'type1'
                np.savez_compressed(os.path.join(self.outdir, 'activations_del_{}_G{}_{}_spike.npz'. \
                         format(type_, G, batch_idx)), **sbuffers)
                print('DELAYED AEDAT SPIKES SAVED')

            vmem += vprev
        # looping over all frame repeats (end)

        return total_event_spikes

    def simulate_delayed_spikes_gated(self, spikenet, in_spikes, num_delayed_layers, batch_idx, target ):
        ''' in_spikes: tensor of shape 1,C,H,W,T             '''
        ''' spikenet: SNN                                    '''
        ''' num_delayed_layers: # layers with delayed spikes '''

        _, C, H, W, T = in_spikes.size()
        total_time = T
        G = self.gate_time

        delayed_model, spikenet_new = self.bifurcate_model(spikenet, num_delayed_layers)
        delayed_model_dict = serialize_model(delayed_model) #list(delayed_layers.keys())
        delayed_model_keys = list(delayed_model_dict)

        ########## process delayed layer (start) ###############
        in_spikes_tchw = in_spikes.permute(0,4,1,2,3).squeeze(0) # T,C,H,W
        #print('in_spikes_tchw ', in_spikes_tchw.size())
        i = 0
        for del_name, layer in delayed_model.named_children():
            if isinstance(layer, (nn.Conv2d, nn.AvgPool2d, nn.Linear)):
                ######################## (inference on delayed layer)
                pre_act = layer(in_spikes_tchw) # T,C,H,W

                if self.simulation_params['measure_sparsity'] or self.simulation_params['record_spikes'] \
                    or self.simulation_params['measure_ops']:
                    dhooks, dbuffers = self.create_buffers(delayed_model, img_size=(1,C,H,W,T), spike=True)
                    # NOTE: creating twice as large buffers in this case to accommodate larger simulation time
                    shooks, sbuffers = self.create_buffers(spikenet_new, img_size=(1,*pre_act.size()[1:],2*T), \
                            spike=True)

            vth = delayed_model_dict[delayed_model_keys[i+1]].threshold
        ########## process delayed layer (end) ###############

        num_slots = int( T / G )
        if num_slots == 0: num_slots = 1
        layer1_out_spikes = torch.zeros((1, *pre_act.size()[1:], 2*T), device=self.device) # 1,C,H,W,2*T
        vmem = 0
        for n in range(num_slots):
            vmem += torch.sum(pre_act[n*G:(n+1)*G,...], axis=0)
            t = n*G + G-1
            layer1_out_spikes[...,t-(G-1)] = torch.where(vmem >= vth, torch.tensor(1, device=self.device), \
                    torch.tensor(0, device=self.device))
            vmem = torch.where(vmem >= vth, vmem - vth, vmem) # soft reset vmem

        # simulating layers 2 to L
        for n in range(num_slots):
            for g in range(G):
                t = n*G + g  # start from the fist firing time
                outspikes = spikenet_new(layer1_out_spikes[...,t])
                if self.fire:
                    total_event_spikes += outspikes
                else:
                    total_event_spikes = outspikes

                if self.simulation_params['record_spikes'] or \
                        (self.simulation_params['measure_sparsity'] and rf == 0) \
                        or self.simulation_params['measure_ops']:
                    dbuffers = self.save_spikes(layer, dhooks, dbuffers, in_spikes[...,t], t)
                    sbuffers = self.save_spikes(spikenet_new, shooks, sbuffers, layer1_out_spikes[...,t], t)

        ########### simulating layers 2 to L until vmem is dissipated ###########
        rem_dur = max_spike_count = int(torch.max(vmem) / vth)
        self.max_rem_dur = max(rem_dur, self.max_rem_dur)
        print('remaining duration: {}+({}), max remaining duration: {}+({})'.format(rem_dur, G, \
                self.max_rem_dur, G))
        for t in range(G*num_slots, G*num_slots + int(rem_dur) ):
            # FIRE!!
            layer1_out_spikes[...,t] = torch.where(vmem >= vth, torch.tensor(1, device=self.device), \
                    torch.tensor(0, device=self.device))
            vmem = torch.where(vmem >= vth, vmem - vth, vmem) # soft reset vmem

            outspikes = spikenet_new(layer1_out_spikes[...,t])
            if self.fire:
                total_event_spikes += outspikes
            else:
                total_event_spikes = outspikes

            if self.simulation_params['record_spikes'] or \
                    (self.simulation_params['measure_sparsity'] and rf == 0) \
                    or self.simulation_params['measure_ops']:
                sbuffers = self.save_spikes(spikenet_new, shooks, sbuffers, layer1_out_spikes[...,t], t)
        ########### simulating layers 2 to L until vmem is dissipated ###########

        # unifying both buffers
        # save to file the DVS spikes created during SNN simulation
        if self.simulation_params['record_spikes'] or \
                (self.simulation_params['measure_sparsity'] and rf == 0) \
                or self.simulation_params['measure_ops']:
            sbuffers['spikerelu1'] = sbuffers.pop('input')
            for k, v in dbuffers.items():
                if k == 'input':
                    sbuffers[k] = dbuffers[k]
            sbuffers = self.sort_dict(sbuffers)

        # measure sparsity of all layers
        if self.simulation_params['measure_sparsity'] and rf == 0:
            self.snn_sparsity, self.snn_prev_data_size = self.measure_sparsity(shooks, sbuffers, \
                    in_spikes, 1, self.snn_sparsity, self.snn_prev_data_size, spike=True)

        # save OPs measured for current batch
        if self.simulation_params['measure_ops']:
            self.save_ops(shooks, sbuffers, in_spikes, 1, target.cpu().item())

        if self.simulation_params['record_spikes']:
            for k, v in sbuffers.items():
                print(k, v.shape, torch.sum(v).item())
                sbuffers[k] = sbuffers[k].cpu().numpy()

            G = self.simulation_params['gate_time']
            if G < 0:
                type_ = ''
            else:
                type_ = 'type1'
            np.savez_compressed(os.path.join(self.outdir, 'activations_del_{}_G{}_{}_spike.npz'. \
                     format(type_, G, batch_idx)), **sbuffers)
            print('DELAYED AEDAT SPIKES SAVED')

        return total_event_spikes


    def simulate_with_aedat_spikes(self, extra_time, test_dataset=None):

        self.RF = RF = self.simulation_params.get('num_frame_repeats', 1)
        MAX_TIME = self.simulation_params['tWindow']
        interleave = self.simulation_params['interleave']
        record_spikes = self.simulation_params['record_spikes']
        record_poisson_spikes = self.simulation_params['record_poisson_spikes']
        num_delayed_layers = self.simulation_params['num_delayed_layers']
        delayed_spikes = self.simulation_params['delayed_spikes']
        delayed_gated = self.simulation_params['delayed_gated']
        save_activation = self.simulation_params['save_activation']
        gate_time = self.simulation_params['gate_time']
        #fire_time = self.simulation_params['fire_time']
        #BATCH_NUM = 1 # (for combined 4 and 6)
        #BATCH_NUM = 25 # (for plain and full aggr)
        #BATCH_NUM = 0 # (for combined all)
        #BATCH_NUM = 2 # (for plain and full aggr)
        BATCH_NUM = 0 # (for 4,6,combined all)

        spikenet_temp = self.convert_to_snn()
        if self.simulation_params['measure_sparsity']:
            self.snn_sparsity = {}
            self.snn_prev_data_size = 0
            self.snn_sparsity['input'] = 0
            for n, m in spikenet_temp.named_modules():
                if isinstance(m, spikeRelu):
                    layer_name = 'spikerelu'+n
                    self.snn_sparsity[layer_name] = 0

        if self.simulation_params['measure_ops']:
            temp = {}
            for c in range(11):
                temp[str(c)] = (0, 0)
            self.spike_rate, self.max_spike_count = {}, {}
            self.spike_rate['input'] = copy.deepcopy( temp)
            self.max_spike_count['input'] = 0
            for n, m in spikenet_temp.named_modules():
               if isinstance(m, spikeRelu):
                    layer_name = 'spikerelu'+n
                    self.spike_rate[layer_name] = copy.deepcopy( temp)
                    self.max_spike_count[layer_name] = 0
        spikenet_temp = None
        self.max_rem_dur = 0

        total_inputs, undef_dvs = 0, 0
        correct_ann, correct_event_snn, correct_poisson_snn, correct_analog_snn = 0, 0, 0, 0
        confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        with torch.no_grad():
            for batch_idx, (input, target, max_val) in enumerate(self.aedatTestLoader):
                print ('\n\n------------ inferring batch {} -------------'.format(batch_idx))

                # input is of shape NCHWT
                input, target, max_val = input.to(self.device), target.to(self.device), max_val.to(self.device)

                ## to control the sampled image (remove later)
                if save_activation or record_spikes or record_poisson_spikes:
                    if not batch_idx == BATCH_NUM:
                        continue

                batch_size, ic, ih, iw, it = input.size()
                time_window = it
                if RF < 0:
                    repeat_factor = int(MAX_TIME / it)
                else:
                    repeat_factor = self.RF

                zeropad = torch.zeros((batch_size, ic, ih, iw, extra_time), device=self.device, dtype=input.dtype)
                print('total spikes: {}, input size: {}'.format( torch.sum(input), input.size()))
                print('Extra time: {}, num Frame repeats: {}, interleave: {}'\
                        .format(extra_time, repeat_factor, interleave))
                print('Threshold percentile: {}'.format(self.percentile))
                print('num frame repeats: {}, extra time: {}'.format(repeat_factor, extra_time))

                # create image from dvs inputs
                print('creating images from dvs spikes...')
                image = torch.sum(input, dim=(-1)) #, keepdim=True).squeeze(-1)
                image /= max_val

                ############### ANN inference (start) ###############
                if save_activation:
                    if len(image.size()) == 4:
                        hooks, buffers = self.create_buffers(self.net, img_size=(1,ic,ih,iw), spike=False)
                    else:
                        hooks, buffers = self.create_buffers(self.net, img_size=(1,ih,iw), spike=False)

                if self.simulation_params['measure_sparsity']:
                    hooks, buffers = self.create_buffers(self.net, img_size=(1,ic,ih,iw), spike=False)
                    self.ann_sparsity = {}
                    self.prev_data_size = 0
                    for name, _ in buffers.items():
                        self.ann_sparsity[name] = 0

                # perform inference with images on ANN
                self.net.eval()
                output_ann = self.net(image.float())
                _, predicted_ann = torch.max(output_ann, 1)
                correct_ann += predicted_ann.eq(target).sum().item()

                if self.simulation_params['measure_sparsity']:
                    self.ann_sparsity, self.prev_data_size = self.measure_sparsity(hooks, buffers, image, \
                            batch_size, self.ann_sparsity, self.prev_data_size)

                if save_activation:
                    if batch_idx == BATCH_NUM:
                        filename_prefix = 'activations_{}'.format( batch_idx )
                        if self.simulation_params['combine_frames']:
                            CF = self.simulation_params['combine_factor']
                            filename_prefix += '_comb' + str(CF)
                        self.save_activations(self.net, hooks, buffers, image, filename_prefix)
                        print('ACTIVATIONS SAVED')
                ############### ANN inference (end) ###############

                ############### SNN inference (start) ###############

                # (1) perform SNN inference on DVS spikes
                # create the spiking model
                spikenet = (self.convert_to_snn()).to(self.device)
                spikenet.eval()

                # create temporary hooks and buffers
                if record_spikes or self.simulation_params['measure_sparsity'] \
                        or self.simulation_params['measure_ops']:
                    #print('target: ', target)
                    shooks, sbuffers = self.create_buffers(spikenet, img_size= \
                            (1, ic, ih, iw, it*self.RF + extra_time), spike=True)

                total_event_spikes = torch.zeros((batch_size, self.num_classes), device=self.device)
                if interleave:
                    new_input = torch.repeat_interleave(input, repeats=repeat_factor, dim=-1) # 1,C,H,W,T
                    assert torch.sum(new_input[...,0:repeat_factor]) == repeat_factor * torch.sum(new_input[...,0])
                else:
                    repeat_along_dims = [1] * (len(input.size())-1)
                    repeat_along_dims.append(repeat_factor)
                    new_input = input.repeat(*repeat_along_dims) # entire frame repeat
                new_input = torch.cat((new_input, zeropad), -1) # append 0s for `zeropad` time

                print('total dvs events: {:.4f} K, dvs input size: {}'.format(torch.sum(new_input) * 1 / 1000, \
                        new_input.size()))

                if delayed_spikes:
                    if delayed_gated:
                        self.gate_time = gate_time
                        #self.type3_fire_time = fire_time
                        print('Delayed + gated ==> gate time: ', self.gate_time)
                        total_event_spikes = self.simulate_delayed_spikes_gated(spikenet, new_input, \
                                num_delayed_layers, batch_idx, target)
                    else:
                        print('Delayed + cyclic ==> gate time = frame duration ')
                        total_event_spikes = self.simulate_delayed_spikes_cyclic(spikenet, new_input, \
                                num_delayed_layers, batch_idx, target)

                else:
                # start of else condition (plain DVS SNN)

                    ############# starting of time-stepped spike integration of (DVS) SNN
                    for t in range(time_window*repeat_factor + extra_time):
                        # inference of spikenet on DVS inputs
                        dvs_spikes = new_input[...,t]
                        start = datetime.now()
                        outspikes = spikenet(dvs_spikes.float())
                        end = datetime.now()
                        #print('dvs inference time: {} ms'.format( (end - start).total_seconds()*1000))
                        if self.fire:
                            total_event_spikes += outspikes
                        else:
                            total_event_spikes = outspikes

                        if record_spikes:
                            if batch_idx == BATCH_NUM:
                                if t == 0:
                                    print('=== Recording spikes ===')
                                self.save_spikes(spikenet, shooks, sbuffers, dvs_spikes, t)

                        if self.simulation_params['measure_sparsity'] or self.simulation_params['measure_ops']:
                            self.save_spikes(spikenet, shooks, sbuffers, dvs_spikes, t)
                    ############## end of time-stepped spike integration (DVS) SNN

                    # save sparsity measured for current batch
                    if self.simulation_params['measure_sparsity']:
                        self.snn_sparsity, self.snn_prev_data_size = self.measure_sparsity(shooks, sbuffers, \
                                new_input, batch_size, self.snn_sparsity, self.snn_prev_data_size, spike=True)

                    # save OPs measured for current batch
                    if self.simulation_params['measure_ops']:
                        self.save_ops(shooks, sbuffers, new_input, batch_size, target.cpu().item())
                # end of else condition (plain DVS SNN)

                if torch.count_nonzero(total_event_spikes) == 0:
                    print('Prediction indeterminate')
                    predicted_event_snn = torch.tensor(-1, device=self.device)
                    undef_dvs += 1
                    # NOTE: DVS non-determinate inputs are counted as wrong inputs
                else:
                    _, predicted_event_snn = torch.max(total_event_spikes.data, -1)
                correct_event_snn += predicted_event_snn.eq(target).sum().item()

                # save to file the DVS spikes created during SNN simulation
                if record_spikes and not self.delayed_spikes:
                    if batch_idx == BATCH_NUM:
                        for k, v in sbuffers.items():
                            print(sbuffers[k].size())
                            sbuffers[k] = sbuffers[k].cpu().numpy()
                        filename_prefix = 'activations_spike{}'.format( batch_idx )
                        if self.simulation_params['combine_frames']:
                            CF = self.simulation_params['combine_factor']
                            filename_prefix += '_comb' + str(CF)
                        np.savez_compressed(os.path.join(self.outdir, filename_prefix+'.npz'), \
                                        **sbuffers)
                        print('AEDAT SPIKES SAVED')

                total_poisson_spikes = torch.zeros((batch_size, self.num_classes), device=self.device)
                if self.simulation_params['measure_ops'] and self.simulation_params['combine_frames'] and \
                        self.simulation_params['combine_factor'] < 0:
                            pass

                else:
                    ## (2) perform SNN inference on Poisson spikes
                    # create the second spiking model
                    spikenet2 = (self.convert_to_snn()).to(self.device)
                    spikenet2.eval()

                    if record_poisson_spikes:
                        sphooks, spbuffers = self.create_buffers(spikenet2, img_size=(1, ic, ih, iw, \
                                it*self.RF + extra_time), spike=True)

                    repeat_along_dims = [1] * (len(image.size()))
                    repeat_along_dims.append(time_window)
                    images_rpt = image.unsqueeze(-1) # 1,C,H,W,1
                    images_rpt = images_rpt.repeat(*repeat_along_dims) # 1,C,H,W,T
                    poisson_spikes  = self.poisson_spikes(images_rpt, device=self.device).float() # 1,C,H,W,T
                    if interleave:
                        poisson_spikes = torch.repeat_interleave(poisson_spikes, repeats=repeat_factor, dim=-1) # interleave (1,C,H,W,T*rpt_fac)
                        assert torch.sum(poisson_spikes[:,:,:,:,0:repeat_factor]) == repeat_factor * torch.sum(poisson_spikes[:,:,:,:,0])
                    else:
                        repeat_along_dims[-1] = repeat_factor
                        poisson_spikes = poisson_spikes.repeat(*repeat_along_dims) # 1,C,H,W,T*rpt_fac, entire frame repeat

                    poisson_spikes = torch.cat((poisson_spikes, zeropad.float()), -1) # zero pad
                    print('total poisson spikes: {:.4f} K, poisson input size: {}'.format( torch.sum(poisson_spikes) / 1000, \
                            poisson_spikes.size()))
                    ############# starting of time-stepped spike integration of (Poisson) SNN
                    for t in range(time_window*repeat_factor + extra_time):
                        start = datetime.now()
                        out2 = spikenet2(poisson_spikes[...,t].float())
                        end = datetime.now()
                        #print('poisson inference time: {} ms'.format((end - start).total_seconds()*1000) )
                        if self.fire:
                            total_poisson_spikes += out2
                        else:
                            total_poisson_spikes = out2

                        if record_poisson_spikes:
                            curr_class = target.cpu().numpy()[0]
                            if batch_idx == BATCH_NUM:
                                if t == 0:
                                    print('=== Recording poisson spikes ===')
                                self.save_spikes(spikenet2, sphooks, spbuffers, poisson_spikes[...,t], t)

                    ############# end of time-stepped spike integration of (Poisson) SNN

                _, predicted_poisson_snn = torch.max(total_poisson_spikes.data, -1)
                correct_poisson_snn += predicted_poisson_snn.eq(target).sum().item()

                # save to file the Poisson spikes created during SNN simulation
                if record_poisson_spikes:
                    if batch_idx == BATCH_NUM:
                        for k, v in spbuffers.items():
                            print(spbuffers[k].size())
                            spbuffers[k] = spbuffers[k].cpu().numpy()
                        np.savez_compressed(os.path.join(self.outdir, 'activations_pspike{}.npz'.format( batch_idx )), \
                                        **spbuffers)
                        print('POISSON SPIKES SAVED')

                # confusion matrix
                for t, p in zip(target.view(-1), predicted_event_snn.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                print('ground truth: ', target.cpu().numpy(), '\nclass (ANN):  ', predicted_ann.cpu().numpy())
                print('class (dvs):  ', predicted_event_snn.cpu().numpy(), '\nclass (Poisson): ', predicted_poisson_snn.cpu().numpy())

                print('Output layer spike count')
                for b in range(batch_size):
                    print('baseline ANN: ', output_ann.cpu().numpy())
                    if len(total_event_spikes.size()) > 1:
                        print('event SNN:    ', total_event_spikes[b,:].cpu().numpy())
                        print('poisson SNN:  ', total_poisson_spikes[b,:].cpu().numpy())
                    else:
                        print('event SNN:    ', total_event_spikes.cpu().numpy())
                        print('poisson SNN:  ', total_poisson_spikes.cpu().numpy())

                total_inputs += target.size(0)
                ann_acc = correct_ann / total_inputs * 100
                event_snn_acc = correct_event_snn / total_inputs * 100
                poisson_snn_acc = correct_poisson_snn / total_inputs * 100
                print('Accuracy on {} inputs: ANN: {:.3f}%  SNN event: {:.3f}%, poisson: {:.3f}%, undef DVS: {}'.\
                        format(total_inputs, ann_acc, event_snn_acc, poisson_snn_acc, undef_dvs))

                if save_activation or record_spikes or record_poisson_spikes:
                    if batch_idx == BATCH_NUM: break
            ##### end of looping over all batches

            if self.simulation_params['measure_sparsity']:
                print('----- ANN activation sparsity -----')
                for k, v in self.ann_sparsity.items():
                    print('{}: {:.3f} %'.format(k, v*100))

                print('----- SNN activation sparsity -----')
                for k, v in self.snn_sparsity.items():
                    print('{}: {:.3f} %'.format(k, v*100))

            if self.simulation_params['measure_ops']:
                self.writeOpsToExcel(sbuffers, delayed_spikes)

            print('Class-wise accuracy with DVS: {}'.format(confusion_matrix.diag()/confusion_matrix.sum(1)))
            print(confusion_matrix.long())


    def simulate_snn(self, test_dataset=None):
        time_window = self.simulation_params['tWindow']
        input_type = self.simulation_params['input_type']
        self.extra_time = self.simulation_params['extra_time']

        # create a spiking model
        spikenet = self.convert_to_snn()
        print(spikenet)
        spikenet.eval()

        if input_type == 'poisson':
            print('-------- simulating with poisson spikes -------- ')
            self.simulate_with_poisson_spikes(spikenet, time_window)
        elif input_type == 'aedat':
            if self.datatype == 'const_time':
                print('-------- simulating with aedat spikes (constant time frames) -------- ')
            else:
                print('-------- simulating with aedat spikes (constant evtCnt frames) -------- ')
            # NOTE: using the same function to simulate both const event and const time frames
            self.simulate_with_aedat_spikes(self.extra_time, test_dataset )

    def plot_spikes(self, samples=[0]):
        for s in samples:
            filenames = ['layerwise_Pspikes{}.npz'.format(s), 'layerwise_Espikes{}.npz'.format(s)]
            for filename in filenames:
                filepath = os.path.join(self.outdir, filename)
                if os.path.exists(filepath):
                    container = np.load(filepath)
                    acts = [container[key] for key in container]
                    for i, a in enumerate(acts):
                        print(a.shape)
                        fpath = os.path.join(self.outdir, filename[10]+'raster_'+str(s)+'_l'+str(i)+'.png')
                        plot_spike_raster(a, fpath)
                else:
                    print('WARNING: {} does not exist!'.format(filename))

