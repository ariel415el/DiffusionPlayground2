import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, transforms


def get_transforms(im_size, center_crop,  gray_scale):
    transforms = [T.ToTensor()]

    if center_crop:
        transforms += [T.CenterCrop(size=center_crop)]
    if gray_scale:
        transforms += [T.Grayscale()]
    if im_size is not None:
        transforms += [T.Resize(im_size, antialias=True),]
    transforms+=[T.Normalize((0.5,), (0.5,))]

    return T.Compose(transforms)


class DiskDataset(Dataset):
    def __init__(self, args):
        super(DiskDataset, self).__init__()
        self.paths = [os.path.join(args.data_root, im_name) for im_name in os.listdir(args.data_root)]
        if args.limit_data is not None:
            self.paths = self.paths[:args.limit_data]
        self.transforms = get_transforms(args.im_size, args.center_crop,  args.gray_scale)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if transforms is not None:
            img = self.transforms(img)
        return img


def create_mnist_dataloaders(args, num_workers=4):
    train_dataset = DiskDataset(args)
    return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
