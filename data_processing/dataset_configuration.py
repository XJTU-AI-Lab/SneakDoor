import torch
import torchvision

from hyperparams.general_params import general_args
from hyperparams.log import logger
from torchvision import transforms
import numpy as np

DATASET_LENGTH_CIFAR10 = 50000

NUM_CLASSES_CIFAR10 = 10

IMAGE_SIZE_CIFAR10 = (32, 32)

def get_dataset_info(dataset):
    data = dict()
    data['name'] = dataset.lower()
    if dataset.lower() == 'cifar10':
        data['dataset_length'] = DATASET_LENGTH_CIFAR10
        data['num_classes'] = NUM_CLASSES_CIFAR10
        data['img_size'] = IMAGE_SIZE_CIFAR10
        data['mean'] = (0.4914, 0.4822, 0.4465)
        data['std'] = (0.2023, 0.1994, 0.2010)
        data['attack_from'] = -1
        data['attack_to'] = 0
    else:
        raise ValueError('Unrecognized Image Dataset !')
    data['min'] = ((np.array([0,0,0]) - np.array(data['mean'])) / np.array(data['std'])).min()
    data['max'] = ((np.array([1,1,1]) - np.array(data['mean'])) / np.array(data['std'])).max()
    if general_args.num_classes < data['num_classes']:
        data['num_classes'] = general_args.num_classes
        logger.warning_once(f"⚠️ Number of classes is set to {general_args.num_classes} for dataset {dataset}")
    return data

def get_dataset_obj(dataset):
    if dataset.lower() == 'cifar10':
        return torchvision.datasets.CIFAR10
    else:
        raise ValueError('Unrecognized Image Dataset !')
    
def get_datasets(name, resize=False):
    if name.lower() in ['cifar10', 'cifar100']:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(get_dataset_info(name)['mean'], get_dataset_info(name)['std']),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(get_dataset_info(name)['mean'], get_dataset_info(name)['std']),
        ])
        train_dataset = get_dataset_obj(name)(
            root='../bench_datasets/image_datasets/', 
            train=True,
            download=True, 
            transform=transform_train
        )
        test_dataset = get_dataset_obj(name)(
            root='../bench_datasets/image_datasets/', 
            train=False,
            download=True, 
            transform=transform_test
        )
    for dataset in train_dataset, test_dataset:
        if hasattr(dataset, 'labels'):
            dataset.targets = dataset.labels
        else:
            dataset.labels = dataset.targets
    # if name in ['mnist', 'fmnist']:
    #     dataset_info = get_dataset_info(name)
    #     if dataset_info['img_size'] != (28,28):
    #         resized_imgs = []
    #         for img in train_dataset.data:
    #             # 转换为 torch.Tensor 并添加批次和通道维度
    #             img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()  # 形状: (1, 1, 28, 28)
    #             # 执行插值
    #             img_resized = F.interpolate(img_tensor, size=dataset_info['img_size'], mode='bilinear', align_corners=False)
    #             # 移除批次和通道维度并转换回 NumPy
    #             img_resized = img_resized.squeeze(0).squeeze(0).numpy()  # 形状: (32, 32)
    #             resized_imgs.append(img_resized)
    #         train_dataset.data = torch.tensor(resized_imgs)
    #         resized_imgs = []
    #         for img in test_dataset.data:
    #             # 转换为 torch.Tensor 并添加批次和通道维度
    #             img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    #             # 执行插值
    #             img_resized = F.interpolate(img_tensor, size=dataset_info['img_size'], mode='bilinear', align_corners=False)
    #             # 移除批次和通道维度并转换回 NumPy
    #             img_resized = img_resized.squeeze(0).squeeze(0).numpy()
    #             resized_imgs.append(img_resized)
    #         test_dataset.data = torch.tensor(resized_imgs)
    return train_dataset, test_dataset