"""
'pytorch_dataset' module contains the implementations of:
- AERRecordingsDataset Class
- AERNumpyDataset Class
- AERNumpyDataset Class
"""
import os
import numpy as np
from torch.utils.data import IterableDataset, Dataset
from dv import AedatFile


class AERRecordingsDataset(IterableDataset):
    """
    This is a pytorch IterableDataset that reads aedat v4 files directly,
    provided they contain pre-accumulated frames. These pre-accumulated frames
    will be read and returned as they are, or together with their label.

    :param data_list_file: A .txt file containing the list of aedat files and \
    their respective labels (separated by a tab).
    :param stream_name: The name of the desired stream in the aedat file.
    :param return_labels: Whether to return the label alongside each frame.
    """

    def __init__(
        self, data_list_file: str, stream_name: str = "frames", return_labels: bool = False,
    ):
        super().__init__()
        self.return_labels = return_labels
        self.stream_name = stream_name

        # read list file
        self.file_list = []
        self.labels_list = []
        with open(os.path.expanduser(data_list_file), "r") as f:
            for line in f:
                url, label = line.split("\t")
                url = os.path.expanduser(url)
                self.file_list.append(url)
                self.labels_list.append(label)

    def __iter__(self):
        for file, label in zip(self.file_list, self.labels_list):
            with AedatFile(file) as f:
                for frame in f[self.stream_name]:
                    if self.return_labels:
                        yield frame.image, label
                    else:
                        yield frame.image


class AERNumpyDataset(Dataset):
    """
    This is a pytorch Dataset that reads individual npz files saved by the
    AedatConvert class's to_numpy method.

    :param file: The path to the npz file.
    :param return_labels: Whether to return the label alongside each frame.
    """

    def __init__(
        self, file: str, return_labels: bool = False,
    ):
        super().__init__()
        self.file = np.load(os.path.expanduser(file))
        self.return_labels = return_labels

    def __len__(self):
        return len(self.file["frames"])

    def __getitem__(self, i: int):
        frame = self.file["frames"][i]
        if self.return_labels:
            return frame, self.file["labels"][i]
        return frame


class AERFolderDataset(Dataset):
    """
    This is a pytorch Dataset that reads multiple npz files organized in
    folders by the AedatConvert class's to_folder method.

    :param root: The path to the folder, which has to contain subfolders \
    for each label value.
    :param transform: A transformation to be applied to each image.
    :param target_transform: A  transformation to be applied to each label.
    :param from_spiketrain: If False (default), return the accumulated frames \
    as they are saved in each file. If True, recompute the frames from the \
    spike trains.
    :param dt: The time interval to use for accumulation if from_spiketrain \
    is set to True.
    :param keep_polarity: If True, separate negative and positive events in two separate channels \
    when from_spiketrain is True.
    :param which: Select data that have this value in the last column of the csv file (e.g. 'test', 'train')
    """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        from_spiketrain=False,
        dt=1000,
        keep_polarity=False,
        which=None,
    ):
        super().__init__()

        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.keep_polarity = keep_polarity
        if from_spiketrain:
            # will reaccumulate 1 ms frames starting from the spiketrain
            self.dt = dt
            self.loader = self._ms_raster_loader
        else:
            # just load the pre-accumulated frames from the file
            self.loader = self._frame_loader

        framelist = np.loadtxt(os.path.join(root, "frames.csv"), delimiter=", ", dtype=str)
        if which is not None:
            idx = framelist[:, -1]  # last column indicates 'test', 'train'...
            framelist = framelist[idx == which]
        self.samples = framelist[:, :2]

    @staticmethod
    def _frame_loader(file):
        with np.load(file) as f:
            frames = f['frame'].astype(np.float32)
        return frames

    @staticmethod
    def _make_raster(spiketrain, dt, bins_xy, keep_polarity=False):
        x, y, t, p = spiketrain["x"], spiketrain["y"], spiketrain["t"], spiketrain["p"]
        timebins = np.arange(t[0], t[-1] + dt + 1, dt)
        timebins[-1] -= 1
        if keep_polarity:
            raster, _ = np.histogramdd((t, p, x, y), (timebins, (-1, 0.5, 2), *bins_xy))
        else:
            raster, _ = np.histogramdd((t, x, y), (timebins, *bins_xy))
            raster = raster[:, np.newaxis]
        return raster.astype(np.float32)

    def _ms_raster_loader(self, file):
        with np.load(file) as f:
            bins_xy = f['bins_xy']
            raster = self._make_raster(f['spiketrain'], self.dt, bins_xy, self.keep_polarity)
        return raster

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(os.path.join(self.root, path))
        #print('original shape ', sample.shape)
        #sample = sample.transpose(1,2,0)
        #print('permuted shape ', sample.shape)
        target = int(target)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
