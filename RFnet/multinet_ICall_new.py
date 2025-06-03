import os
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights
from torchvision.transforms import Pad
from pytorch_toolbelt.losses import CrossEntropyFocalLoss
import timm
from PIL import Image
import scipy.io as sio
import scipy.interpolate
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # Import tqdm
import os
import torchvision.transforms.functional as F
import torch.nn.functional as F2
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Custom Dataset
class MultiModalDataset(Dataset):
    def __init__(self, image_dir, feature_dir, iq_dir, transform=None, modalities='image', feature_size=8, iq_size=16384, da='snr_freshift', pre=False, valid=False):
        self.image_dir = image_dir
        self.feature_dir = feature_dir
        self.iq_dir = iq_dir
        self.transform = transform
        self.modalities = modalities
        self.pre = pre
        self.valid = valid
        if self.pre:
            # 测试集包含所有以 .png 结尾的文件
            self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        else:
            pattern_freshift = re.compile(r'.*DA1-.*\.png$')
            pattern_no = re.compile(r'.*DA1-.*_DA1\.png$')
            match da:
                case 'snr':  # 作为对比方案验证数据增强效果时选择
                    # 训练集只包含以 _DA1~3.png 结尾的文件
                    self.image_files = [f for f in os.listdir(image_dir) if f.endswith('_DA1.png') or f.endswith('_DA2.png') or f.endswith('_DA3.png')] # or f.endswith('_DA2.png') or f.endswith('_DA3.png')]
                case 'snr_freshift':  # 其它情况选择
                    # 训练集包含所有以 .png 结尾的文件
                    self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
            self.image_files = [
                f for f in self.image_files
                if self._get_label_from_filename(f) in range(0, 5)  # 作为对比方案训练时改为range(0, 10)，作为识别器训练时为range(0, 5)
            ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        feature_name = img_name.replace('OneUAVImage', 'OneUAVFeat').replace('.png', '.mat')
        feature_path = os.path.join(self.feature_dir, feature_name)
        iq_name = img_name.replace('OneUAVImage', 'OneUAVIQ').replace('.png', '.mat')
        iq_path = os.path.join(self.iq_dir, iq_name)

        data = {}

        if 'image' in self.modalities:
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            data['image'] = image

        if self.pre:
            label = torch.tensor(-1)
        else:
            label = self._get_label_from_filename(img_name)  # - 5  # Adjust label to start from 0
            # if label > 0:
            #     label = 1

        return data, label, img_name
    @staticmethod
    def _get_label_from_filename(filename):
        label_match = re.search(r'_([0-9]+)_snr', filename)
        if label_match:
            return int(label_match.group(1))
        return None

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels_list, SA=False):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        self.reduce_convs = nn.ModuleList()
        self.SA = SA
        if SA:
            self.attentions = nn.ModuleList()

        for in_channels, out_channels in zip(in_channels_list, out_channels_list):
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            self.reduce_convs.append(nn.Conv2d(out_channels, out_channels // 2, 1))  # 新增的卷积层，用于减少通道数
            if SA:
                self.attentions.append(EMA(out_channels))

        self.top_down = nn.Conv2d(out_channels_list[-1], out_channels_list[-1], 1)

    def forward(self, x):
        # Bottom-up pathway
        c1, c2, c3, c4 = x

        # Top-down pathway
        p4 = self.lateral_convs[3](c4)
        p4_upsampled = F2.interpolate(p4, scale_factor=2, mode='nearest')
        p3 = self.lateral_convs[2](c3) + p4_upsampled
        p3_upsampled = F2.interpolate(p3, scale_factor=2, mode='nearest')
        p2 = self.lateral_convs[1](c2) + p3_upsampled
        p2_upsampled = F2.interpolate(p2, scale_factor=2, mode='nearest')
        p1 = self.lateral_convs[0](c1) + p2_upsampled

        # Top-down convs
        p1 = self.output_convs[0](p1)
        p2 = self.output_convs[1](p2)
        p3 = self.output_convs[2](p3)
        p4 = self.output_convs[3](p4)

        if self.SA:
            # Apply attention
            p1 = self.attentions[0](p1)
            p2 = self.attentions[1](p2)
            p3 = self.attentions[2](p3)
            p4 = self.attentions[3](p4)

        return p1, p2, p3, p4


class CustomNet(nn.Module):
    def __init__(self, num_classes, feature_size, iq_size, modalities='image', net='ResNet', pre=False, use_fpn=False, use_ema=False):
        super(CustomNet, self).__init__()
        self.modalities = modalities
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.iq_size = iq_size
        self.netType = net
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))  # 添加自适应池化层
        self.use_fpn = use_fpn
        self.use_ema = use_ema

        if 'image' in modalities:
            if self.netType == 'ResNet':
                if pre:
                    weight = None
                else:
                    weight = ResNet50_Weights.IMAGENET1K_V2
                self.net_image = resnet50(weights=None)  # weights=ResNet50_Weights.IMAGENET1K_V2
                self.net_image.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                # self.net_image.conv1 = nn.Conv2d(1, 64, kernel_size=(14, 4), stride=(4, 1), padding=(6, 0), bias=False)
                self.net_image.fc = nn.Identity()

                # Extract features from different layers
                self.layer1 = self.net_image.layer1
                self.layer2 = self.net_image.layer2
                self.layer3 = self.net_image.layer3
                self.layer4 = self.net_image.layer4

                # Add EMA to each layer if use_ema is True
                self.ema1 = EMA(256) if use_ema else nn.Identity()
                self.ema2 = EMA(512) if use_ema else nn.Identity()
                self.ema3 = EMA(1024) if use_ema else nn.Identity()
                self.ema4 = EMA(2048) if use_ema else nn.Identity()

                # Add FPN with attention
                if use_fpn and use_ema:
                    self.fpn = FPN(in_channels_list=[256, 512, 1024, 2048], out_channels_list=[512, 512, 512, 512], SA=False)
                elif use_fpn:
                    self.fpn = FPN(in_channels_list=[256, 512, 1024, 2048], out_channels_list=[512, 512, 512, 512], SA=False)
                elif use_ema:
                    self.fpn = None
                else:
                    self.fpn = None

        combined_size = 0
        if 'image' in modalities:
            if self.netType == 'ResNet':
                combined_size += 2048
            elif self.netType == 'Vit':
                combined_size += 768  # ViT base 模型的输出尺寸
            elif self.netType == 'DRNN':
                combined_size += 2048  # DRNN 模型魔改的输出尺寸
        if 'feature' in modalities:
            combined_size += 256
        if 'iq' in modalities:
            combined_size += 256

        self.bn = nn.BatchNorm1d(combined_size)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(combined_size, num_classes)

    def forward(self, image=None, feature=None, iq=None, pre=False):
        x = []
        if 'image' in self.modalities and image is not None:
            if self.netType == 'ResNet':
                # Process image through ResNet layers
                c1 = self.layer1(self.net_image.maxpool(self.net_image.relu(self.net_image.bn1(self.net_image.conv1(image)))))
                c1SA = self.ema1(c1)

                c2 = self.layer2(c1)
                c2SA = self.ema2(c2)

                c3 = self.layer3(c2)
                c3SA = self.ema3(c3)

                c4 = self.layer4(c3)
                c4SA = self.ema4(c4)

                # Process through FPN
                if self.fpn is not None:
                    if self.use_ema:
                        p1, p2, p3, p4 = self.fpn([c1SA, c2SA, c3SA, c4SA])
                    else:
                        p1, p2, p3, p4 = self.fpn([c1, c2, c3, c4])
                    p1_flat = p1.mean(dim=[2, 3])
                    p2_flat = p2.mean(dim=[2, 3])
                    p3_flat = p3.mean(dim=[2, 3])
                    p4_flat = p4.mean(dim=[2, 3])

                    x.append(torch.cat([p1_flat, p2_flat, p3_flat, p4_flat], dim=1))
                elif self.use_ema:
                    x.append(c4SA.mean(dim=[2, 3]))
                else:
                    x.append(c4.mean(dim=[2, 3]))

        if 'feature' in self.modalities and feature is not None:
            x.append(feature)

        if 'iq' in self.modalities and iq is not None:
            x.append(iq)

        if len(x) > 0:
            x = torch.cat(x, dim=1)
            # x = self.bn(x)  # 添加批量归一化层
            x = self.dropout(x)
            x = self.classifier(x)
        else:
            raise ValueError("No input provided for any modality")
        return x


# Training function
def train(model, dataloader, criterion, optimizer, device, modalities):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, labels, image_names in tqdm(dataloader, desc="Training", leave=False):
        inputs = {}
        for modality in modalities:
            inputs[modality] = data.get(modality).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total
    return running_loss / len(dataloader), acc

def validate(model, dataloader, criterion, device, modalities):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels, image_names in tqdm(dataloader, desc="Validating", leave=False):
            inputs = {}
            for modality in modalities:
                inputs[modality] = data.get(modality).to(device)
            labels = labels.to(device)

            outputs = model(**inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return running_loss / len(dataloader), acc

def predict(models, dataloader, device, modalities, outfilepath):
    # Test the models
    results = []
    all_labels = []
    all_preds = []
    # 将所有模型切换到评估模式
    for model in models:
        model.eval()
    with torch.no_grad():
        for data, labels, image_names in tqdm(dataloader, desc="Testing", leave=False):
            inputs = {}
            for modality in modalities:
                inputs[modality] = data.get(modality).to(device)
            labels = labels.to(device)

            outputs = [model(**inputs, pre=False) for model in models]
            outputs = torch.stack(outputs).mean(dim=0)  # Average predictions
            _, predicted = torch.max(outputs, 1)

            for img_name, label, pred in zip(image_names, labels, predicted):
                results.append((img_name, pred.item()))
                all_labels.append(label.item())
                all_preds.append(pred.item())

    # Save results to file
    with open(outfilepath, 'w') as f:
        for img_name, pred in results:
            f.write(f'{img_name} {pred}\n')


def sample_dataset(dataset, num_samples):
    indices = list(range(len(dataset)))
    sampled_indices = random.choices(indices, k=num_samples)
    return torch.utils.data.Subset(dataset, sampled_indices)


def pad_to_target_size(img, target_height, target_width):
    current_width, current_height = img.size
    # print(f'currentheight {current_height}, current_width {current_width}')  # current width=1024?
    if current_width > 224:
        img1 = img
        img2 = img1.resize((224, 224), Image.Resampling.BILINEAR)  # 使用双线性插值进行缩放
    else:
        padding_top = max(0, (target_height - current_height) // 2)
        padding_bottom = max(0, target_height - current_height - padding_top)
        padding_left = max(0, (target_width - current_width) // 2)
        padding_right = max(0, target_width - current_width - padding_left)
        padding = (padding_left, padding_top, padding_right, padding_bottom)
        img1 = Pad(padding)(img)
        img2 = img1.resize((224, 224), Image.Resampling.BILINEAR)  # 使用双线性插值进行缩放
    width, height = img2.size
    # print(f'trans_height {height}, trans_width {width}')
    return img2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(mode='train', modalities='image', da='no', numclass=5, netType='ResNet', num_bags=10, fpn=False, ema=False):
    # 设置随机种子
    set_seed(666)
    # 选择模态组合
    modalities = modalities  # 可根据需要调整 ['image', 'feature', 'iq']
    print(f'Using modalities: {modalities}')
    # 路径设置
    train_image_dir = "D:\download\\UAVDataset\\rectrain_yolo\images"
    # "D:\download\\UAVDataset\\rectrain"
    # 'D:\download\\UAVDataset\\rectrain_yolo\\images'
    train_feature_dir = 'D:/download/UAV-Sigmf-float16/UAVsigmf-npdata/rectrain'
    train_iq_dir = 'D:/download/UAV-Sigmf-float16/UAVsigmf-npdata/rectrain'
    valid_image_dir = "D:\download\\UAVDataset\\recvalid_yolo\\images"
    # "D:\download\\UAVDataset\\recvalid"
    # 'D:\download\\UAVDataset\\recvalid_yolo\\images'
    valid_feature_dir = 'D:/download/UAV-Sigmf-float16/UAVsigmf-npdata/recvalid'
    valid_iq_dir = 'D:/download/UAV-Sigmf-float16/UAVsigmf-npdata/recvalid'
    test_image_dir = 'D:/download/UAVDataset\\alltest_fixsnr\\yoloimg'
    # 'D:/download/UAVDataset\\alltest'
    # 'D:\download\\UAVDataset\\alltest\\yoloimg'
    test_feature_dir = 'D:/download/UAV-Sigmf-float16/UAVsigmf-npdata/rectest'
    test_iq_dir = 'D:/download/UAV-Sigmf-float16/UAVsigmf-npdata/rectest'

    # Parameters
    num_classes = numclass  # Adjust based on your dataset!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    feature_size = 8  # Adjust based on your feature vector size
    iq_size = 16384  # 根据实际数据调整 2^14 1024*16
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.0005  # resnet 0.0005
    weight_decay = 0.00005  # resnet 0.00005
    match da:
        case 'snr':
            model_path = 'uav_rec_lsnr_ICallDA2/uav_rec_lsnr_all_snr2DAimg_resnet'
            outfilepath = 'uav_rec_lsnr_ICallDA2/result_decrec_snr2DAimg_all_resnet.txt'
        case 'snr_freshift':
            model_path = 'uav_rec_lsnr_ICallDA2/uav_rec_lsnr_ICpart_snr2DAimg_resnetyolodata224'
            # uav_rec_lsnr_ICpart_snr2DAimg_resnetyolodata224
            # uav_rec_lsnr_ICpart_snr2DAimg_resnet(FPN+EMA)yolodata224
            outfilepath = 'uav_rec_lsnr_ICallDA2/result_lsnr_ICpart_fixsnr_resnetyolodata224(top10).txt'
            # result_lsnr_ICpart_snr2DAimg_resnetyolodata224pro(2).txt
            # result_lsnr_ICpart_snr2DAimg_resnet(FPN+EMA)yolodata224(20).txt
    # 检查并创建文件夹
    model_dir = os.path.dirname(model_path)
    outfile_dir = os.path.dirname(outfilepath)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(outfile_dir, exist_ok=True)

    if netType == 'ResNet':
        transform = transforms.Compose([
            # transforms.Lambda(lambda img: pad_to_target_size(img, 512, 224)),
            # transforms.CenterCrop((224, 224)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mode == 'train':
        # Datasets and Dataloaders
        train_dataset = MultiModalDataset(train_image_dir, train_feature_dir, train_iq_dir, transform=transform,
                                          modalities=modalities, feature_size=feature_size, iq_size=iq_size, da=da)
        valid_dataset = MultiModalDataset(valid_image_dir, valid_feature_dir, valid_iq_dir, transform=transform,
                                          modalities=modalities, feature_size=feature_size, iq_size=iq_size,
                                          valid=True)
        # Sample datasets for bagging
        if num_bags == 0:
            bagged_train_datasets = [train_dataset]
        else:
            bagged_train_datasets = [sample_dataset(train_dataset, len(train_dataset)) for _ in range(num_bags)]
        bagged_train_loaders = [DataLoader(bagged_train_datasets[i], batch_size=batch_size, shuffle=True) for i in
                                range(num_bags)]
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # Dictionary to store the best validation accuracy for each bag
        best_valid_accs = {}

        # Training loop
        for bag_idx in range(num_bags):
            # Early stopping parameters
            patience = 10
            best_valid = 0.0
            patience_counter_valid = 0

            model = CustomNet(num_classes=num_classes, feature_size=feature_size, iq_size=iq_size,
                              modalities=modalities, net=netType, use_fpn=fpn, use_ema=ema).to(device)
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            for epoch in range(num_epochs):
                train_loss, train_acc = train(model, bagged_train_loaders[bag_idx], criterion, optimizer, device, modalities)
                valid_loss, valid_acc = validate(model, valid_loader, criterion, device, modalities)
                print(
                    f'Bag {bag_idx + 1}/{num_bags}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    torch.save(model.state_dict(), f"{model_path}_bag{bag_idx}.pth")
                    patience_counter_valid = 0  # Reset the counter if we get a new best model
                else:
                    patience_counter_valid += 1
                # Early stopping check
                if patience_counter_valid >= patience:  # or patience_counter_valid
                    print("Early stopping triggered.")
                    break
            # Store the best validation accuracy for this bag
            best_valid_accs[bag_idx] = best_valid

        # Sort the bags by their best validation accuracy in descending order
        sorted_bags = sorted(best_valid_accs.items(), key=lambda item: item[1], reverse=True)
        # Save the top 10 models
        top_n = 10
        for i in range(min(top_n, num_bags)):
            bag_idx, best_valid_acc = sorted_bags[i]
            print(f'Saving Bag {bag_idx + 1} with Best Valid Acc: {best_valid_acc:.4f}')
            torch.save(torch.load(f"{model_path}_bag{bag_idx}.pth"), f"{model_path}_top{bag_idx}.pth")

    elif mode == 'predict':
        # Load the models
        models = [
            CustomNet(num_classes=num_classes, feature_size=feature_size, iq_size=iq_size, modalities=modalities,
                      net=netType, pre=True, use_fpn=fpn, use_ema=ema).to(device) for _ in range(num_bags)]
        # models2 = [
        #     CustomNet(num_classes=num_classes, feature_size=feature_size, iq_size=iq_size, modalities=modalities,
        #               net='Vit').to(device) for _ in range(num_bags)]
        # models = models1 + models2
        for bag_idx in range(num_bags):
            models[bag_idx].load_state_dict(torch.load(f"{model_path}_top{bag_idx}.pth"))  # 预测看这行,半数最优改为top
            # models[bag_idx+5].load_state_dict(torch.load(f"{model_path2}_bag{bag_idx}.pth"))
        # Datasets and Dataloaders
        test_dataset = MultiModalDataset(test_image_dir, test_feature_dir, test_iq_dir, transform=transform,
                                         modalities=modalities, feature_size=feature_size, iq_size=iq_size,
                                         pre=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        # Test the model
        predict(models, test_loader, device, modalities, outfilepath)


if __name__ == '__main__':
    # 选择模态组合
    modalities = ['image']  # 可根据需要调整 ['image', 'feature', 'iq']
    # Change to 'predict' for testing  # da=['snr' 'freshift' 'snr_freshift' 'no']  # netType='ResNet','Vit'
    main(mode='predict', modalities=modalities, da='snr_freshift', numclass=5, netType='ResNet', num_bags=10, fpn=False, ema=False)

    # 在验证Bagging集成学习方法时，设置num_bags=10，代表有10个子学习器。需要验证子学习器数目对性能影响时
