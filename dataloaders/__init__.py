from dataloaders.datasets import pascal
from dataloaders.datasets import cityscapes
from dataloaders.datasets import coco
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == "cityscapes":
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
        test_loader = None
        
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
         train_set = coco.COCOSegmentation(args, split='train')
         val_set = coco.COCOSegmentation(args, split='val')
         num_class = train_set.NUM_CLASSES
         train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
         val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
         test_loader = None
         return train_loader, val_loader, test_loader, num_class
        
    else:
        raise NotImplementedError

