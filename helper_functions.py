import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import kornia as K
import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage import rotate as scipyrotate
from networks import MLP, ConvNet, ResNet, VGG, BasicBlock
import pandas as pd
from os import walk
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


def get_network(model, channel, num_classes, im_size, dist=True):

    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
            net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
        
    elif model == 'ResNet':
        net = ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes)

    elif model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)

    elif model == 'VGG':
        net = VGG('VGG16', channel, num_classes)

    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num>0:
            device = 'cuda'
            if gpu_num>1:
                net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

    return net


def get_time():
    return str(time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime()))


class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return self.images.shape[0]
    


def epoch(mode, dataloader, net, optimizer, criterion, args, aug, texture=False):
    loss_avg, acc_avg, num_exp = 0,0,0
    net = net.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)

        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)

        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)

        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg

    
    
def get_deparam(dataset, model, model_eval, ipc):

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    return dc_aug_param
    


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5

        

def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1

        

def DiffAugment(x, strategy='', seed=-1, param=None):
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x
    
    # if strategy:
    #     if param.aug_mode == 'A':
    #         for p in strategy.split('_'):
    #             for f in AUGMENT_FNS[p]:
    #                 x = f(x, param)
    #     elif param.aug_mode == 'S':
    #         pbties = strategy.split('_')
    #         set_seed_DiffAug(param)
    #         p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
    #         for f in AUGMENT_FNS[p]:
    #             x = f(x, param)
    #     else:
    #         exit('Error ZH: unknown augmentation mode.')
    #     x = x.contingous()

    return x



def augment(images, dc_aug_param, device):

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1], shape[2]+crop*2, shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)

        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images

def get_eval_pool(eval_mode, model, model_eval):
    # All models
    if eval_mode == "A":
        model_eval_pool = ["ConvNet", "MLP", "ResNet"]
    
    # Model plus ResNet
    elif eval_mode == "R":
        model_eval_pool = [model, "ResNet"]

    # Model plus MLP
    elif eval_mode == "M":
        model_eval_pool = [model, "MLP"]
    
    # Model plus ConvNet
    elif eval_mode == "C":
        model_eval_pool[model, "ConvNet"]

    # Just the model itself
    elif eval_mode == "S":
        model_eval_pool = [model_eval]

    else:
        model_eval_pool = [model_eval]

    return model_eval_pool

def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, return_loss=False, texture=False):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=True, texture=texture)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)


    time_train = time.time() - start

    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test

# AUGMENT_FNS = {
#     'color': [rand_brightness, rand_saturation, rand_contrast],
#     'crop': [rand_crop],
#     'cutout': [rand_cutout],
#     'flip': [rand_flip],
#     'scale': [rand_scale],
#     'rotate': [rand_rotate],
# }

def get_full_dataset(dataset, data_path, batch_size=1, args=None):
    
    class_map = None
    loader_train_dict = None
    class_map_inv = None

    if dataset == 'EuroSAT':
        channel = 3
        im_size = (64, 64)
        num_classes = 10
        mean = [0.344, 0.380, 0.407]
        std = [0.202, 0.136, 0.115]

        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else: 
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        # Define some variables
        IDX_CLASS_LABELS = {
            0: 'AnnualCrop',
            1: 'Forest',
            2: 'HerbaceousVegetation',
            3: 'Highway',
            4: 'Industrial',
            5: 'Pasture',
            6: 'PermanentCrop',
            7: 'Residential',
            8: 'River',
            9: 'SeaLake'
        }
        CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture','PermanentCrop','Residential','River', 'SeaLake']
        CLASS_IDX_LABELS = dict()

        for key, val in IDX_CLASS_LABELS.items():
            CLASS_IDX_LABELS[val] = key
        
        VALID_SIZE = 0.1

        FULL_DATA_DF = os.path.join(data_path, 'FULL_DATA.csv')

        if os.path.exists(FULL_DATA_DF):
            # Read from existing DF
            DATA_DF = pd.read_csv(FULL_DATA_DF)

        else:
            # Read all data files into DF
            i = 0
            DATA_DF = pd.DataFrame(columns=['image_id', 'label'])

            for (dirpath, dirname, filename) in walk(data_path):
                for each_file in filename:
                    DATA_DF.loc[i] = [each_file, dirpath.split('/')[-1]]
                    i += 1

            DATA_DF.to_csv(FULL_DATA_DF, index=False)

        DATA_DF = DATA_DF.sample(frac=1, random_state=42)
        TRAIN_DF = DATA_DF[:-int(len(DATA_DF)*VALID_SIZE)]
        VALID_DF = DATA_DF[-int(len(DATA_DF)*VALID_SIZE):]

        TRAIN_DF.reset_index(inplace=True)
        VALID_DF.reset_index(inplace=True)

        dst_train = EuroSAT(TRAIN_DF, data_path, transform)
        dst_test = EuroSAT(VALID_DF, data_path, transform)

        class_names = CLASSES
        class_map = {x:x for x in range(num_classes)}

    testloader = DataLoader(dst_test, batch_size=batch_size, shuffle=True, num_workers=2)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv



# EuroSAT Class
class EuroSAT(Dataset):
    def __init__(self, train_df, train_dir, transform=None):
        self.train_dir = train_dir
        self.train_df = train_df
        self.transform = transform

    def __len__(self):
        return len(self.train_df)
    
    def __getitem__(self, index):
        row = self.train_df.loc[index]
        img_id, label = row['image_id'], row['label']
        img = Image.open(os.path.join(self.train_dir, img_id.split('.')[0].split('_')[0], img_id))
        if self.transform:
            img = self.transform(img)
        return img, encode_label(label)

# Define some variables
IDX_CLASS_LABELS = {
    0: 'AnnualCrop',
    1: 'Forest',
    2: 'HerbaceousVegetation',
    3: 'Highway',
    4: 'Industrial',
    5: 'Pasture',
    6: 'PermanentCrop',
    7: 'Residential',
    8: 'River',
    9: 'SeaLake'
}
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture','PermanentCrop','Residential','River', 'SeaLake']
CLASS_IDX_LABELS = dict()

for key, val in IDX_CLASS_LABELS.items():
    CLASS_IDX_LABELS[val] = key

# Give idx of each class name
def encode_label(label):
    idx = CLASS_IDX_LABELS[label]
    return idx

# Take in idx and return the class name
def decode_target(target, text_labels=True):
    if text_labels:
        return IDX_CLASS_LABELS[target]
    else:
        return target


