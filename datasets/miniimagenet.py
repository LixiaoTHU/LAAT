import os
from torch.utils import data
from PIL import Image


class MiniImageNet(data.Dataset):
    """MiniImageNet few-shot dataset.

    Parameters
    ----------
    root: string
        Root directory including `miniimagenet` subdirectory.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: Callable
        A (series) of valid transformation(s).
    target_transform: Callable
        A (series) of valid transformation(s) on target.
    """

    base_folder = "miniimagenet"
    url = "https://drive.google.com/open?id=1R6dA6QGEW-lmiNkitCwK4IkAbl4uT3y3"

    def __init__(self, root, split="train", transform=None, target_transform=None):
        self.root = os.path.join(root, self.base_folder)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        split_file = os.path.join(
            self.root, "splits", "ravi-larochelle", self.split + ".txt"
        )
        classlist = []
        classmap = {}
        with open(split_file) as f:
            for line in f.readlines():
                classlist.append(line.rstrip())
        with open(os.path.join(self.root, "map_clsloc.txt")) as f:
            for line in f:
                cls, loc, name = line.rstrip().split()
                name = name.replace("_", " ")
                classmap[cls] = name
        self.classlist = list(map(lambda _: classmap[_], classlist))

        self.imgs, self.ids = [], []
        for i in range(len(classlist)):
            folder = os.path.join(self.root, "data", classlist[i])
            names = os.listdir(folder)
            for n in names:
                imgpath = os.path.join(folder, n)
                self.imgs.append(imgpath)
                self.ids.append(i)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.open(self.imgs[i])
        target = self.ids[i]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


if __name__ == "__main__":
    from torchvision import transforms
    from samplers import NShotTaskSampler

    root = os.path.expanduser("~/data1/dataset")
    base_transform = transforms.Compose(
        [transforms.Lambda(lambda _: _.convert("RGB")), transforms.ToTensor()]
    )
    trainset = MiniImageNet(root, split="train", transform=base_transform)
    valset = MiniImageNet(root, split="val", transform=base_transform)
    testset = MiniImageNet(root, split="test", transform=base_transform)
    trainloader = data.DataLoader(trainset, batch_size=40, shuffle=True, num_workers=4)
    print("Train:")
    for i, batch in enumerate(trainloader):
        imgs, cids = batch
        print(imgs.shape)
        print(cids)
        if i == 2:
            break
    valsampler = NShotTaskSampler(valset.ids, 5, 2, 20)
    valloader = data.DataLoader(
        valset, sampler=valsampler, batch_size=65536, num_workers=2
    )
    print("Val:")
    p = 5 * 2
    for t in range(3):
        print(f"repeat {t}:")
        for i, batch in enumerate(valloader):
            print("classes:", valsampler.get_class_labels(valset.classlist))
            if i == 0:
                imgs, cids = batch
                X_s, X_q = imgs[:p], imgs[p:]
                y_s, y_q = cids[:p], cids[p:]
                print("support:")
                print(X_s.shape)
                print(valsampler.convert_target(y_s))
                print("query:")
                print(X_q.shape)
                print(valsampler.convert_target(y_q))
            else:
                imgs, cids = batch
                print(imgs.shape)
                print(cids, valsampler.convert_target(cids))
    # test is same as val
    testsampler = NShotTaskSampler(testset.ids, 5, 2, 20)
    testloader = data.DataLoader(testset, sampler=testsampler, batch_size=65536)
    print("Test:")
    for t in range(3):
        print(f"repeat {t}:")
        for i, batch in enumerate(testloader):
            print("classes:", testsampler.get_class_labels(testset.classlist))
            if i == 0:
                imgs, cids = batch
                X_s, X_q = imgs[:p], imgs[p:]
                y_s, y_q = cids[:p], cids[p:]
                print("support:")
                print(X_s.shape)
                print(testsampler.convert_target(y_s))
                print("query:")
                print(X_q.shape)
                print(testsampler.convert_target(y_q))
            else:
                imgs, cids = batch
                print(imgs.shape)
                print(testsampler.convert_target(cids))
