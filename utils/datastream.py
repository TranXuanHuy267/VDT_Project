from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from torch.utils.data.distributed import DistributedSampler

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(267)

class Imdataset(Dataset):
    def __init__(self, instances, transform=None):
        super().__init__()
        self.instances = instances
        if transform is not None:
            self.instances = [(transform(image), label) for image, label in instances] 
    
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

    def collate_fn(self, batch):
        images = []
        labels = []
        for inst in batch:
            images.append(inst[0])
            labels.append(inst[1])
        images = torch.tensor(images)
        labels = torch.tensor(labels)
        return [images, labels]

def collect_images(task, split_ratio, number_id):
    dataset = []
    assert task in ["Train", "Valid", "Test"]
    folder_dir = "WebFace260M"
    assert number_id <= len(os.listdir(folder_dir))
    ids = os.listdir(folder_dir)[:number_id]
    id_to_label = dict(zip(ids, range(len(ids))))
    for id in tqdm(os.listdir(folder_dir)[:number_id]):
        if id.startswith("0_0_"):
            id_images = os.listdir(folder_dir + "/" + id)
            id_images_num = len(id_images)
            test_num = int(split_ratio["Test"]*id_images_num)
            valid_num = int(split_ratio["Valid"]*id_images_num)
            if task=="Train":
                images = id_images[:id_images_num - valid_num - test_num]
            elif task=="Valid":
                images = id_images[id_images_num - valid_num - test_num:id_images_num - test_num]
            else:
                images = id_images[id_images_num - test_num:]

            for images in images:
                if images.endswith(".jpg"):
                    dataset.append(tuple([Image.open(folder_dir + "/" + id + "/" + images).convert('RGB'), id_to_label[id]]))

    return dataset

def prepare_data(number_id=10, batch_size=4, num_workers=2, split_ratio=None, train_sample_size=None, valid_sample_size=None, test_sample_size=None, multigpu=False):
        
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = Imdataset(collect_images("Train", split_ratio, number_id), train_transform)
    if train_sample_size is not None:
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)
    
    if multigpu:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                                        shuffle=False, sampler=DistributedSampler(trainset))
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                                            shuffle=True, num_workers=num_workers,)
    
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    validset = Imdataset(collect_images("Valid", split_ratio, number_id), test_transform)
    
    if valid_sample_size is not None:
        indices = torch.randperm(len(validset))[:valid_sample_size]
        validset = torch.utils.data.Subset(validset, indices)
    
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, pin_memory=True,
                                            shuffle=False, num_workers=num_workers)


    testset = Imdataset(collect_images("Test", split_ratio, number_id), test_transform)
    
    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, pin_memory=True,
                                            shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader