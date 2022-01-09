import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import numpy as np

# from IPython.core.debugger import Tracer
# debug_here = Tracer()


def default_loader(path):
    return Image.open(path).convert('RGB')


class Sh_Views_Dataset(data.Dataset):
    def __init__(self, data_folder, shape_flist, transform=None, loader=default_loader):
        """
        data_folder: [str] directory to view images 
        shape_flist: [str] path to a text file. Each line is a (filename, class) pair. e.g.
            example_%d.jpg 0
            example2_%d.jpg 1

        transform: the transformation applied to an input image
        """
        self.data_folder = data_folder
        self.sample_pairs = [
            line.strip().split(' ')  # parse each line in the text file
            for line in open(shape_flist).readlines()]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        sh_path = self.sample_pairs[index][0]
        sh_view_imgs = []
        #print(sh_path)
        for i in range(13):
            sh_img = self.loader(os.path.join(self.data_folder, sh_path))
            if self.transform is not None:
                sh_img = self.transform(sh_img)
            sh_view_imgs.append(sh_img)

        sh_imgs = torch.stack(sh_view_imgs)

        if len(self.sample_pairs[index]) >= 2:
            target = int(self.sample_pairs[index][1])
        else:   # in case the label is not given, e.g., during testing
            target = -1
        return sh_imgs, target

    def __len__(self):
        return len(self.sample_pairs)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from misc import transforms as T
    sh_file = 'shrec19/train/lists/model_train_pair.txt'
    print('testing')
    # debug_here()
    # default_flist_reader('./merged_views_2d_sample', sh_file)
    train_transformer = T.Compose([
        # T.RandomSizedRectCrop(height, width),
        T.RectScale(224, 224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # normalizer,
    ])
    sh_views_dataset = Sh_Views_Dataset('shrec19/train/Views', sh_file, transform=train_transformer)
    imgs, label = sh_views_dataset[0]
    print(imgs.shape, label)
    print('done')
