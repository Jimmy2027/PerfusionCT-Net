import os
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from mlebe.training.utils import data_loader as dl
from mlebe.training.utils.general import preprocess
from dataio.loaders.utils import validate_images
import torch.utils.data as data
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from .utils import validate_images


class mlebe_dataset(Dataset):

    def __init__(self, template_dir, data_dir, studies, split, transform=None,
                 split_seed=42, train_size=0.7, test_size=0.15, valid_size=0.15):
        """
        if train_size = None, no splitting of the data is done
        """
        super(mlebe_dataset, self).__init__()
        self.data_selection = self.make_dataselection(data_dir, studies)
        self.transform = transform
        self.template_dir = template_dir
        self.split = split

        if train_size:
            test_valid_size = test_size + valid_size

            train_selection, test_val_selection = train_test_split(self.data_selection, train_size=train_size,
                                                                   test_size=test_valid_size,
                                                                   random_state=split_seed)
            test_selection, validation_selection = train_test_split(test_val_selection,
                                                                    train_size=test_size / test_valid_size,
                                                                    test_size=valid_size / test_valid_size,
                                                                    random_state=split_seed)

            if split == 'train':
                self.selection = train_selection
            if split == 'test':
                self.selection = test_selection
            if split == 'validation':
                self.selection = validation_selection

        else:
            self.selection = self.data_selection

        self.ids = self.selection['uid'].to_list()

    def make_dataselection(self, data_dir, studies):
        data_selection = pd.DataFrame()
        for o in os.listdir(data_dir):
            if (o in studies or not studies) and not o.startswith('.') and not o.endswith(
                    '.xz'):  # i.e. if o in studies or if studies empty
                print(o)
                data_set = o
                for x in os.listdir(os.path.join(data_dir, o)):
                    if x.endswith('preprocessing') or x.startswith('preprocess') and not x.endswith('work'):
                        for root, dirs, files in os.walk(os.path.join(data_dir, o, x)):
                            for file in files:
                                if file.endswith("_T2w.nii.gz") or file.endswith("_T1w.nii.gz"):
                                    split = file.split('_')
                                    subject = split[0].split('-')[1]
                                    session = split[1].split('-')[1]
                                    acquisition = split[2].split('-')[1]
                                    type = split[3].split('.')[0]
                                    uid = file.split('.')[0]
                                    path = os.path.join(root, file)
                                    data_selection = pd.concat([data_selection, pd.DataFrame(
                                        [[data_set, subject, session, acquisition, type, uid, path]],
                                        columns=['data_set', 'subject', 'session', 'acquisition', 'type', 'uid',
                                                 'path'])])
        return data_selection

    def get_ids(self, indices):
        return [self.ids[index] for index in indices]

    def __len__(self):
        return len(self.selection)

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        img = nib.load(self.selection.iloc[index]['path']).get_data()
        target = dl.load_mask(self.template_dir).get_data()

        img = preprocess(img, (128, 128), 'coronal')
        target = preprocess(target, (128, 128), 'coronal')

        # Make sure there is a channel dimension
        img = np.expand_dims(img, axis=-1)
        target = np.expand_dims(target, axis=-1)

        # handle exceptions
        validate_images(img, target)

        # apply transformations
        if self.transform:
            transformer = self.transform()
            img, target = transformer(img, target)

        return img, target, index
