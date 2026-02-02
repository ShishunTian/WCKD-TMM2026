import os
from ImageDataset import ImageDataset, TID_ImageDataset
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from torchvision import transforms

from PIL import Image

from args import Configs

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

live_train_csv = os.path.join('./IQA_Database/databaserelease2/splits2', '7', 'live_train_clip.txt')
live_val_csv = os.path.join('./IQA_Database/databaserelease2/splits2', '7', 'live_val_clip.txt')
live_test_csv = os.path.join('./IQA_Database/databaserelease2/splits2', '7', 'live_test_clip.txt')

csiq_train_csv = os.path.join('./IQA_Database/CSIQ/splits2', '7', 'csiq_train_clip.txt')
csiq_val_csv = os.path.join('./IQA_Database/CSIQ/splits2', '7', 'csiq_val_clip.txt')
csiq_test_csv = os.path.join('./IQA_Database/CSIQ/splits2', '7', 'csiq_test_clip.txt')

bid_train_csv = os.path.join('./IQA_Database/BID/splits2', '7', 'bid_train_clip.txt')
bid_val_csv = os.path.join('./IQA_Database/BID/splits2', '7', 'bid_val_clip.txt')
bid_test_csv = os.path.join('./IQA_Database/BID/splits2', '7', 'bid_test_clip.txt')

clive_train_csv = os.path.join('./IQA_Database/ChallengeDB_release/splits2', '7', 'clive_train_clip.txt')
clive_val_csv = os.path.join('./IQA_Database/ChallengeDB_release/splits2', '7', 'clive_val_clip.txt')
clive_test_csv = os.path.join('./IQA_Database/ChallengeDB_release/splits2', '7', 'clive_test_clip.txt')

koniq10k_train_csv = os.path.join('./IQA_Database/koniq-10k/splits2', '7', 'koniq10k_train_clip.txt')
koniq10k_val_csv = os.path.join('./IQA_Database/koniq-10k/splits2', '7', 'koniq10k_val_clip.txt')
koniq10k_test_csv = os.path.join('./IQA_Database/koniq-10k/splits2', '7', 'koniq10k_test_clip.txt')

kadid10k_train_csv = os.path.join('./IQA_Database/kadid10k/splits2', '7', 'kadid10k_train_clip.txt')
kadid10k_val_csv = os.path.join('./IQA_Database/kadid10k/splits2', '7', 'kadid10k_val_clip.txt')
kadid10k_test_csv = os.path.join('./IQA_Database/kadid10k/splits2', '7', 'kadid10k_test_clip.txt')

tid_train_csv = os.path.join('./IQA_Database/TID2013/splits2', '7', 'tid_train_score.txt')
tid_test_csv = os.path.join('./IQA_Database/TID2013/splits2', '7', 'tid_test.txt')

config = Configs()

live_set = config.datapath
csiq_set = config.datapath
bid_set = config.datapath
clive_set = config.datapath
koniq10k_set = config.datapath
kadid10k_set = config.datapath
tid_set = config.datapath



def set_dataset(csv_file, bs, data_set, num_workers, preprocess, num_patch, test):
    data = ImageDataset(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=False, num_workers=num_workers)
    return loader


def set_tid_dataset(csv_file, bs, data_set, num_workers, preprocess, num_patch, test):
    data = TID_ImageDataset(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=False, num_workers=num_workers)
    return loader

class AdaptiveResize(object):
    """Resize the input PIL Image to the given size adaptively.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation
        if image_size is not None:
            self.image_size = image_size
        else:
            self.image_size = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size

        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return transforms.Resize(self.image_size, self.interpolation)(img)

        if h < self.size or w < self.size:
            return img
        else:
            return transforms.Resize(self.size, self.interpolation)(img)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _preprocess2():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def _preprocess3():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_data(i):
    preprocess2 = _preprocess2()
    preprocess3 = _preprocess3()
    train_patch = config.train_patch
    batch = config.batch_size

    live_train_loader = set_dataset(live_train_csv, batch, live_set, 8, preprocess3, train_patch, False)
    live_val_loader = set_dataset(live_val_csv, 16, live_set, 8, preprocess2, 15, True)
    live_test_loader = set_dataset(live_test_csv, 16, live_set, 8, preprocess2, 15, True)

    csiq_train_loader = set_dataset(csiq_train_csv, batch, csiq_set, 8, preprocess3, train_patch, False)
    csiq_val_loader = set_dataset(csiq_val_csv, 16, csiq_set, 8, preprocess2, 15, True)
    csiq_test_loader = set_dataset(csiq_test_csv, 16, csiq_set, 8, preprocess2, 15, True)

    bid_train_loader = set_dataset(bid_train_csv, batch, bid_set, 8, preprocess3, train_patch, False)
    bid_val_loader = set_dataset(bid_val_csv, 16, bid_set, 8, preprocess2, 15, True)
    bid_test_loader = set_dataset(bid_test_csv, 16, bid_set, 8, preprocess2, 15, True)

    clive_train_loader = set_dataset(clive_train_csv, batch, clive_set, 8, preprocess3, train_patch, False)
    clive_val_loader = set_dataset(clive_val_csv, 16, clive_set, 8, preprocess2, 15, True)
    clive_test_loader = set_dataset(clive_test_csv, 16, clive_set, 8, preprocess2, 15, True)

    koniq10k_train_loader = set_dataset(koniq10k_train_csv, batch, koniq10k_set, 8, preprocess3, train_patch, False)
    koniq10k_val_loader = set_dataset(koniq10k_val_csv, 16, koniq10k_set, 8, preprocess2, 15, True)
    koniq10k_test_loader = set_dataset(koniq10k_test_csv, 16, koniq10k_set, 8, preprocess2, 15, True)

    kadid10k_train_loader = set_dataset(kadid10k_train_csv, batch, kadid10k_set, 8, preprocess3, train_patch, False)
    kadid10k_val_loader = set_dataset(kadid10k_val_csv, 16, kadid10k_set, 8, preprocess2, 15, True)
    kadid10k_test_loader = set_dataset(kadid10k_test_csv, 16, kadid10k_set, 8, preprocess2, 15, True)

    tid_train_loader = set_tid_dataset(tid_train_csv, batch, tid_set, 8, preprocess3, train_patch, False)
    tid_val_loader = set_tid_dataset(tid_test_csv, 16, tid_set, 8, preprocess2, 15, True)
    tid_test_loader = set_tid_dataset(tid_test_csv, 16, tid_set, 8, preprocess2, 15, True)

    train_loaders = [live_train_loader, csiq_train_loader, bid_train_loader, clive_train_loader, koniq10k_train_loader, kadid10k_train_loader, tid_train_loader]
    val_loaders = [live_val_loader, csiq_val_loader, bid_val_loader, clive_val_loader, koniq10k_val_loader, kadid10k_val_loader, tid_val_loader]
    test_loaders = [live_test_loader, csiq_test_loader, bid_test_loader, clive_test_loader, koniq10k_test_loader, kadid10k_test_loader, tid_test_loader]

    return train_loaders[i], val_loaders[i], test_loaders[i]
