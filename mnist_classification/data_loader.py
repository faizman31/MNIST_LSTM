import torch

from torch.utils.data import DataLoader,Dataset

class MNISTDataset(Dataset):

    def __init__(self,data,labels,flatten=True):
        self.data=data
        self.labels=labels
        self.flatten=flatten

        super().__init__()

    
    def __len__(self):
        return self.data.shape[0]

    
    def __getitem__(self,idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.flatten==True:
            x = x.reshape(-1)

        return x,y


def load_mnist(is_train=True,flatten=True):
    from torchvision import datasets,transforms

    dataset = datasets.MNIST(
        '../data/',train=is_train,download=True,
        transform =transforms.Compose([
            transforms.ToTensor()
        ])
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.reshape(x.shape[0],-1)
    
    return x,y


def get_loaders(config):
    x,y = load_mnist(is_train=True,flatten=False)

    train_cnt = int(x.shape[0] * config.train_ratio)
    valid_cnt = x.shape[0] - train_cnt

    flatten = True if config.model == 'fc' else False

    indices = torch.randperm(x.shape[0])
    train_x,valid_x = torch.index_select(x,dim=0,index=indices).split([train_cnt,valid_cnt],dim=0)
    train_y,valid_y = torch.index_select(y,dim=0,index=indices).split([train_cnt,valid_cnt],dim=0)

    train_loader = DataLoader(
        dataset = MNISTDataset(train_x,train_y,flatten=flatten),
        batch_size = config.batch_size,
        shuffle= True
    )
    valid_loader = DataLoader(
        dataset = MNISTDataset(valid_x,valid_y,flatten=flatten),
        batch_size = config.batch_size,
        shuffle = True
    )

    test_x,test_y = load_mnist(is_train=False,flatten=flatten)

    test_loader = DataLoader(
        dataset = MNISTDataset(test_x,test_y),
        batch_size = config.batch_size,
        shuffle = False
    )

    return train_loader,valid_loader,test_loader
