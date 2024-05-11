import os

class Path(object):

    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/user01/Amir/Knowledge-Distillation/VOCdevkit/VOC2012'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/Cityspaces'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/kaggle/input/coco-2017-dataset/coco2017/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
