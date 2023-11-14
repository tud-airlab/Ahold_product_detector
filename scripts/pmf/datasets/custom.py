from os import listdir

import cv2
import numpy as np
import torchvision.transforms as transforms


def get_mean_pixel_values(dataset_dirs: list):
    """
    Get mean and std pixel values of a dataset
    """
    pixel_values = []
    std_values = []

    for dir in dataset_dirs:
        for class_label in listdir(dir):
            images = listdir(dir + class_label)
            for image in images:
                if image.endswith(".png") or image.endswith(".jpg") or image.endswith(".jpeg"):
                    img = cv2.imread(dir + class_label + "/" + image)
                    pixel_values.append(np.mean(img, axis=(0, 1)))
                    std_values.append(np.std(img, axis=(0, 1)))

    pixel_values = np.asarray(pixel_values)
    pixel_values = np.mean(pixel_values, axis=0)
    std_values = np.asarray(std_values)
    std_values = np.mean(std_values, axis=0)

    return pixel_values, std_values


def dataset_setting(nSupport, img_size=80):
    """
    Return dataset setting

    :param int nSupport: number of support examples
    """
    trainDir = './data/PMF_dataset/train/'
    valDir = './data/PMF_dataset/validation/'
    testDir = './data/PMF_dataset/test/'
    episodeJson = './data/PMF_dataset/val1000Episode_5_way_1_shot.json' if nSupport == 1 \
        else './data/PMF_dataset/val1000Episode_5_way_5_shot.json'

    # print("getting mean pixel values")
    # mean, std = get_mean_pixel_values([trainDir, valDir, testDir])
    # print("retrieved mean pixel values")
    # mean = [x / 255.0 for x in mean]
    # std = [x / 255.0 for x in std]
    # print("mean =", *mean)
    # print("std = ", *std)
    mean = [0.4416, 0.4286, 0.4058]
    std = [0.2010, 0.1880, 0.2029]
    normalize = transforms.Normalize(mean=mean, std=std)
    trainTransform = transforms.Compose(
        [transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
         transforms.RandomPerspective(distortion_scale=0.2, p=.2),
         transforms.RandomRotation(degrees=(-30, 30)),
         transforms.ColorJitter(brightness=.5, hue=.4),
         transforms.ToTensor(),
         normalize
         ])

    valTransform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor(),
                                       normalize])

    inputW, inputH, nbCls = img_size, img_size, 64

    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, episodeJson, nbCls
