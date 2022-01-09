import torch.utils.data as data
import random
from PIL import Image
import os
import os.path
import numpy as np

# from IPython.core.debugger import Tracer
# debug_here = Tracer()


def default_loader(path):
    return Image.open(path).convert('RGB')


class Sk_Dataset(data.Dataset):
    '''Read each (image, label) pair'''
    def __init__(self, data_folder, sketch_flist, transform=None, loader=default_loader, ext=None):
        """
        data_folder: to view images 
        sketch_flist: path to a text file where each line is a (img_name, class_id) pair. e.g.
          img1.png 0
          img2.png 1
          img3.jpg 1
        transforms: transformation applied to input images
        loader: the image loader
        ext: not used
        """
        self.data_folder = data_folder
        self.sample_pairs = [
            line.strip().split(' ')
            for line in open(sketch_flist).readlines()]
        self.transfrom = transform
        self.loader = loader

    def __getitem__(self, index):
        sk_fname = self.sample_pairs[index][0]
        sk_img = self.loader(os.path.join(self.data_folder, sk_fname))
        if self.transfrom is not None:
            sk_img = self.transfrom(sk_img)
        
        if len(self.sample_pairs[index]) >= 2:
            target = int(self.sample_pairs[index][1])
        else:   # in case the label is not given, e.g., during testing
            target = -1
        return sk_img, target

    def __len__(self):
        return len(self.sample_pairs)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from misc import transforms as T
    #sk_file = '../demo_data/lists/sketch_train_pair.txt'
    sk_file = 'shrec19/train/lists/sketch_train_pair.txt'
    print('testing')
    # debug_here()
    # default_flist_reader('./Sketches', sk_file)
    train_transformer = T.Compose([
        # T.RandomSizedRectCrop(height, width),
        T.RectScale(224, 224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # normalizer,
    ])
    #sk_dataset = Sk_Dataset('../demo_data/Sketches', sk_file, transform=train_transformer, ext='.jpg')
    sk_dataset = Sk_Dataset('shrec19/train/Sketches', sk_file, transform=train_transformer, ext='.png')
    img, label = sk_dataset[0]
    print(img.shape, label)
    print('done')
