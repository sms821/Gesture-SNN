import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")

import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import slayerSNN as snn
import myFuncs
import torch.nn as nn
import h5py
from PIL import Image
from torchvision import transforms
import argparse
import numpy as np
from torch.utils.data import IterableDataset, Dataset

class GestureDataset(Dataset):
    """
    This is a pytorch Dataset that reads multiple npz files organized in
    folders by the AedatConvert class's to_folder method.

    :param root: The path to the folder, which has to contain subfolders \
    for each label value.
    :param transform: A transformation to be applied to each image.
    :param from_spiketrain: If False (default), return the accumulated frames \
    as they are saved in each file. If True, recompute the frames from the \
    spike trains.
    """

    def __init__(
        self,
        root,
        sampleFile,
        transform=None,
        from_spiketrain=False,
        dt = 1,
        keep_polarity=False,
        resolution=64,
    ):
        super().__init__()

        self.transform = transform
        self.root = root
        self.from_spiketrain = from_spiketrain
        self.keep_polarity = keep_polarity
        self.res = resolution
        if from_spiketrain:
            # load the spiketrain from file
            self.dt = dt
            self.loader = self._ms_raster_loader
        else:
            # just load the pre-accumulated frames from the file
            self.loader = self._frame_loader

        if self.transform:
            self.augment = self._augment_data

        self.samples = np.loadtxt(sampleFile, dtype=str)
        prefix = 'test'
        if 'train' in sampleFile:
            prefix = 'train'
        self.root += '/' + prefix

    @staticmethod
    def _augment_data(frame):
        """ frame shape: CHW """
        _, H, W = frame.shape
        xs = 4
        ys = 4
        th = 10
        xjitter = np.random.randint(2 * xs) - xs
        yjitter = np.random.randint(2 * ys) - ys
        ajitter = (np.random.rand() - 0.5) * th / 180 * 3.141592654
        sinTh = np.sin(ajitter)
        cosTh = np.cos(ajitter)

        nonzero = np.argwhere(frame > 0)
        x = nonzero[:, 2].astype(int)
        y = nonzero[:, 1].astype(int)
        x_new = (x * cosTh - x * sinTh + xjitter).astype(int)
        y_new = (y * sinTh + y * cosTh + yjitter).astype(int)
        x_new = np.clip(x_new, 0, W-1)
        y_new = np.clip(y_new, 0, H-1)
        outframe = np.zeros(frame.shape)
        outframe[:, y_new, x_new] = frame[:, y, x]
        return outframe

    @staticmethod
    def _frame_loader(file):
        with np.load(file) as f:
            frame = f['frame'].astype(np.float32)
        return frame

    @staticmethod
    def _make_raster(spiketrain, dt, bins_xy, keep_polarity=False):
        x, y, t, p = spiketrain[1,:], spiketrain[0,:], spiketrain[3,:], spiketrain[2,:]
        nc = 1
        if keep_polarity:
            nc = 2
        tmin, tmax = t.min(), t.max()
        if t[0] != tmin or t[-1] != tmax: # or int(tmax-tmin) > 1000:
            dummy = np.zeros((nc,self.res,self.res,int(tmax-tmin)))
            print('[WARNING] dummy input ', dummy.shape)
            return dummy

        timebins = np.arange(t[0], t[-1] + dt + 1, dt)
        if keep_polarity:
            raster, _ = np.histogramdd(( p, y, x, t), ((-1, 0.5, 2), *bins_xy, timebins ))
        else:
            raster, bins = np.histogramdd((y, x, t), (*bins_xy, timebins ))
            raster = np.expand_dims(raster, axis=0)
        return raster.astype(np.float32)

    def _ms_raster_loader(self, file):
        with np.load(file) as f:
            bins_xy = f['bins_yx']
            raster = self._make_raster(f['spiketrain'], self.dt, bins_xy, self.keep_polarity)
        return raster

    def __getitem__(self, index):
        filenm, target = self.samples[index]
        sample = self.loader(os.path.join(self.root, filenm+'.npz'))
        target = int(target)

        if self.transform is not None and self.transform:
            sample = self._augment_data(sample)

        sample = torch.from_numpy(sample)

        if self.from_spiketrain:
            return sample, target, 1.0
        return sample, target

    def __len__(self):
        return len(self.samples)

# Define the network (64x64 images)
class Network2_64(torch.nn.Module):
    def __init__(self, num_channels=1):
        super(Network2_64, self).__init__()
        # define network functions
        self.conv1 = nn.Conv2d(num_channels, 16, 3, padding=1, stride=2, bias=False) # 16,32,32
        self.drop1 = nn.Dropout(0.1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=2, bias=False) # 32,16,16
        self.drop2 = nn.Dropout(0.1)
        self.relu2 = nn.ReLU()

        self.pool3 = nn.AvgPool2d(2) # 32,8,8
        self.drop3 = nn.Dropout(0.1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(32, 64, 3, padding=1, stride=1, bias=False) # 64,8,8
        self.drop4 = nn.Dropout(0.1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1, stride=1, bias=False) # 64,8,8
        self.drop5 = nn.Dropout(0.1)
        self.relu5 = nn.ReLU()

        self.pool6 = nn.AvgPool2d(2) # 64,4,4
        self.drop6 = nn.Dropout(0.4)

        self.fc8   = nn.Linear(1024, 11, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.drop2(x)
        x = self.relu2(x)

        x = self.pool3(x)
        x = self.drop3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.drop4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.drop5(x)
        x = self.relu5(x)

        x = self.pool6(x)
        x = self.drop6(x)

        x = x.view(x.shape[0], -1)

        x = self.fc8(x)
        return x

# Define the network (64x64 images) : has conv->relu->avg->relu chain
class Network2_64_deep(torch.nn.Module):
    def __init__(self, num_channels=1):
        super(Network2_64_deep, self).__init__()
        # define network functions
        self.conv1 = nn.Conv2d(num_channels, 16, 3, padding=1, stride=2, bias=False) # 16,32,32
        self.drop1 = nn.Dropout(0.1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=1, bias=False) # 32,32,32
        self.drop2 = nn.Dropout(0.1)
        self.relu2 = nn.ReLU()

        self.pool3 = nn.AvgPool2d(2) # 32,16,16
        self.drop3 = nn.Dropout(0.1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(32, 64, 3, padding=1, stride=1, bias=False) # 64,16,16
        self.drop4 = nn.Dropout(0.1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1, stride=1, bias=False) # 64,16,16
        self.drop5 = nn.Dropout(0.1)
        self.relu5 = nn.ReLU()

        self.pool6 = nn.AvgPool2d(2) # 64,8,8
        self.drop6 = nn.Dropout(0.1)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(64, 32, 3, padding=1, stride=2, bias=False) # 32,4,4
        #self.drop7 = nn.Dropout(0.35)
        #self.drop7 = nn.Dropout(0.38)
        #self.drop7 = nn.Dropout(0.40)
        #self.drop7 = nn.Dropout(0.41)
        self.drop7 = nn.Dropout(0.42)
        #self.drop7 = nn.Dropout(0.45)
        #self.drop7 = nn.Dropout(0.46)
        self.relu7 = nn.ReLU()

        self.fc8   = nn.Linear(512, 11, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.drop2(x)
        x = self.relu2(x)

        x = self.pool3(x)
        x = self.drop3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.drop4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.drop5(x)
        x = self.relu5(x)

        x = self.pool6(x)
        x = self.drop6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.drop7(x)
        x = self.relu7(x)
        x = x.view(x.shape[0], -1)

        x = self.fc8(x)
        return x

if __name__ == '__main__':
    # Read the network params from yaml file
    parser = argparse.ArgumentParser(description='Deep Learning SNN simulation')
    parser.add_argument('--config-file')
    args = parser.parse_args()
    print (args)
    netParams = snn.params(args.config_file)


    # Define the cuda device to run the code on.
    device = netParams['device']

    appname = netParams['appname']
    outdir = netParams['appname']
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    batch_size = netParams['training']['batch_size']
    num_epochs = netParams['training']['num_epochs']

    # Create network instance.
    num_channels = 1
    if netParams['dataset']['keep_polarity']:
        num_channels = 2
    if 'deep' in appname:
        net = Network2_64_deep(num_channels).to(device).float()
    else:
        net = Network2_64(num_channels).to(device).float()
    print(net)

    # Define training settings
    lr = netParams['training']['lr']
    criterion = nn.CrossEntropyLoss()

    #optimizer = torch.optim.Adam(net.parameters(), lr = lr, amsgrad = True, weight_decay=1e-5)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr, amsgrad = True, weight_decay=5e-4)
    #optimizer = torch.optim.Adam(net.parameters(), lr = lr, amsgrad = True, weight_decay=10e-4)
    #optimizer = torch.optim.Adam(net.parameters(), lr = lr, amsgrad = True, weight_decay=1e-4)
    #optimizer = torch.optim.Adam(net.parameters(), lr = lr, amsgrad = True, weight_decay=3e-4)
    #optimizer = torch.optim.Adam(net.parameters(), lr = lr, amsgrad = True, weight_decay=5e-4)
    #optimizer = torch.optim.Adam(net.parameters(), lr = lr, amsgrad = True, weight_decay=10e-4)
    #optimizer = torch.optim.Adam(net.parameters(), lr = lr, amsgrad = True, weight_decay=12e-4)
    #optimizer = torch.optim.Adam(net.parameters(), lr = lr, amsgrad = True, weight_decay=0)

    # load the dataset
    datasetPath     = netParams['training']['data_path']['in']
    sampleFileTrain = netParams['training']['data_path']['train']
    sampleFileTest  = netParams['training']['data_path']['test']
    transform       = True
    res             = netParams['dataset']['resolution']
    keep_polarity   = netParams['dataset']['keep_polarity']

    trainingSet = GestureDataset (
            root            = datasetPath,
            sampleFile      = sampleFileTrain,
            transform       = transform,
            from_spiketrain = False,
            keep_polarity   = keep_polarity,
            resolution      = res,
            )

    testingSet = GestureDataset (
            root            = datasetPath,
            sampleFile      = sampleFileTest,
            transform       = False,
            from_spiketrain = False,
            keep_polarity   = keep_polarity,
            resolution      = res,
            )
    print('Size of dataset ==> train: {}, test: {}'.format(len(trainingSet), len(testingSet)))
    #myFuncs.visualize_data(trainingSet, outdir, prefix='train', fac=2000)
    #myFuncs.visualize_data(testingSet, outdir, prefix='test', fac=200)
    #exit()

    trainLoader = DataLoader(dataset=trainingSet, batch_size=batch_size, shuffle=True, num_workers=8)
    testLoader  = DataLoader(dataset=testingSet, batch_size=batch_size, shuffle=False, num_workers=8)

    args = {'net': net, 'device': device, 'trainloader': trainLoader, 'testloader': testLoader, \
            'optimizer': optimizer, 'criterion': criterion, 'lr': netParams['training']['lr'], \
            'chkpt_dir': outdir, 'chkpt_filename': netParams['appname'], 'num_epochs': num_epochs, \
            'resume': netParams['training']['resume'], 'num_classes': 11, 'interval_display': True, \
            'save_checkpoint': True, 'criterion_test': None }

    #print(device)
    solver = myFuncs.Solver(**args)
    from torchsummary import summary
    summary(net.cpu(), input_size=(num_channels, 64, 64), device='cpu')
    if netParams['training']['eval']:
        solver.load_model()
        solver.test(get_stats=netParams['training']['get_stats'])
    elif netParams['training']['learn']:
        solver.train()

    # ANN to SNN conversion
    aedatTestLoader = None
    if netParams['ann_to_snn']['simulation_params']['input_type'] == 'aedat':
        aedatTestingSet = GestureDataset (
                root            = datasetPath,
                from_spiketrain = True,
                sampleFile      = sampleFileTest,
                keep_polarity   = netParams['dataset']['keep_polarity'],
                dt              = 1,
                resolution      = res,
        )

        aedatTestLoader = torch.utils.data.DataLoader(
            aedatTestingSet, batch_size=1, shuffle=False, num_workers=1
        )


    myFuncs.load_checkpoint(net, model_path=outdir, file_name=netParams['appname']+'.pt', device=device)
    snn_args = {'net': net, 'device': device, 'dataloader': trainLoader, 'criterion': criterion, \
            'outdir': outdir, 'len_dataset': len(trainingSet), 'valLoader': testLoader, 'batchsize': batch_size, \
            'simulation_params': netParams['ann_to_snn']['simulation_params'], 'num_classes': 11, \
            'aedatTestLoader': aedatTestLoader, 'percentile': netParams['ann_to_snn']['percentile'], \
            'aedatTestLoaderRaw': None, 'classToEventCount': None, 'fire_last_layer': netParams['ann_to_snn']['fire_last_layer'],}

    converter = myFuncs.SNNConverter(**snn_args)
    new_net = None
    if netParams['ann_to_snn']['merge_BN']:
        net_nobn = Network_nobn()
        converter.merge_bn(net_nobn)
        net_nobn.to(device)
        converter.net = net_nobn # updating net with net_nobn
        new_net = net_nobn
        print(net_nobn)
        #solver.test(model=net_nobn, epoch=None)

    if netParams['ann_to_snn']['compute_th']:
        if new_net is None:
            converter.net = net
        converter.compute_thresholds(new_net)
    if netParams['ann_to_snn']['convert']:
        print('\n[INFO] Converting ANN to SNN..')
        spikenet = converter.convert_to_snn()
        print(spikenet)
        out = spikenet(torch.zeros((1,num_channels,res,res), device=device))
    if netParams['ann_to_snn']['simulate']:
        converter.simulate_snn()
    if netParams['show_feature_maps']:
        myFuncs.show_feature_maps(outdir)
    if netParams['show_raster_plots']:
        myFuncs.show_spikes(outdir)
