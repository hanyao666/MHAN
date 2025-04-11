import os
import sys
from tqdm import tqdm
import argparse
import numpy as np
from torchvision.transforms import RandomApply
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets

from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import itertools
import torch.nn.functional as F
from networks.backbone import MHAN

from sklearn.metrics import confusion_matrix
from sam import SAM

eps = sys.float_info.epsilon


class SmoothCrossEntropy(nn.Module):
    """
    loss = SmoothCrossEntropy()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    """

    def __init__(self, alpha=0.1):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
                       (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        batch_size = features.size(0)
        features = features.view(batch_size, -1)
        centers_batch = self.centers.index_select(0, labels)
        loss = (features - centers_batch).pow(2).sum() / 2.0 / batch_size
        return loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='../data/rafdb/', help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=16, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=80, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    return parser.parse_args()


class AttentionLoss(nn.Module):
    def __init__(self):
        super(AttentionLoss, self).__init__()

    def forward(self, x):
        num_head = len(x)
        if num_head < 2:
            # 如果注意力头数量小于2，则损失为0
            return torch.tensor(0.0, device=x[0].device, requires_grad=True)

        loss = 0
        cnt = 0
        for i in range(num_head - 1):
            for j in range(i + 1, num_head):
                mse = F.mse_loss(x[i], x[j])
                loss += mse
                cnt += 1
        # 返回所有注意力头之间MSE损失的均值
        return loss / cnt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j] * 100, fmt) + '%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.tight_layout()


# class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = MHAN(num_class=7, num_head=args.num_head)
    model.to(device)

    # data_transforms = transforms.Compose([
    #     transforms.Resize((112, 112)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomApply([
    #         transforms.RandomRotation(5),
    #         transforms.RandomCrop(112, padding=8)
    #     ], p=0.5),  # 调整了 p=0.2 为 p=0.5
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    #     transforms.RandomErasing(scale=(0.02, 0.25)),
    # ])
    # data_transforms = transforms.Compose([
    #     transforms.Resize((112, 112)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomChoice([
    #         transforms.RandomRotation(5),
    #         transforms.RandomCrop(112, padding=8)
    #     ]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     transforms.RandomErasing(scale=(0.02, 0.25)),
    # ])
    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([  # 使用 RandomApply
            transforms.RandomRotation(5),
            transforms.RandomCrop(112, padding=8)
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])

    train_dataset = datasets.ImageFolder(f'{args.raf_path}/train', transform=data_transforms)
    print('训练集大小:', len(train_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(f'{args.raf_path}/val', transform=data_transforms_val)
    print('验证集大小:', len(val_dataset))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    criterion_cls = SmoothCrossEntropy()  # 使用 SmoothCrossEntropy 作为损失函数
    criterion_at = AttentionLoss()
    # center_loss = CenterLoss(num_classes=7, feat_dim=512, device=device)  # 确保传递 device 参数

    optimizer = SAM(model.parameters(), torch.optim.Adam, lr=args.lr, rho=0.05, adaptive=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for imgs, targets in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)

            out, feat, heads = model(imgs)

            loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads)

            loss.backward()
            optimizer.first_step(zero_grad=True)

            imgs = imgs.to(device)
            targets = targets.to(device)

            out, feat, heads = model(imgs)

            loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads)

            optimizer.zero_grad()
            loss.backward()
            optimizer.second_step(zero_grad=True)

            running_loss += loss.item()  # 使用 .item() 获取标量值
            _, predicts = torch.max(out, 1)
            correct_sum += torch.eq(predicts, targets).sum().item()

        acc = correct_sum / len(train_dataset)
        running_loss /= iter_cnt
        tqdm.write(
            f'[Epoch {epoch}] 训练准确率: {acc:.4f}. 损失: {running_loss:.3f}. 学习率: {optimizer.param_groups[0]["lr"]:.6f}')

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0

            y_true = []
            y_pred = []

            model.eval()
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)

                out, feat, heads = model(imgs)
                loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads)

                running_loss += loss.item()  # 使用 .item() 获取标量值

                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets).sum().item()
                bingo_cnt += correct_num
                sample_cnt += imgs.size(0)

                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())

                if iter_cnt == 0:
                    all_predicted = predicts
                    all_targets = targets
                else:
                    all_predicted = torch.cat((all_predicted, predicts), 0)
                    all_targets = torch.cat((all_targets, targets), 0)
                iter_cnt += 1

            running_loss /= iter_cnt
            scheduler.step()

            acc = bingo_cnt / sample_cnt
            acc = np.around(acc, 4)
            best_acc = max(acc, best_acc)

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

            tqdm.write(
                f"[Epoch {epoch}] 验证准确率: {acc:.4f}. 平衡准确率: {balanced_acc:.4f}. 损失: {running_loss:.3f}")
            tqdm.write(f"最佳准确率: {best_acc}")

            if acc > 0.92 and acc == best_acc:
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join('checkpoints',
                                        f"rafdb_epoch{epoch}_acc{acc:.4f}_bacc{balanced_acc:.4f}.pth"))
                tqdm.write('模型已保存。')

                # 计算混淆矩阵并保存
                matrix = confusion_matrix(all_targets.cpu().numpy(), all_predicted.cpu().numpy())
                np.set_printoptions(precision=2)
                plt.figure(figsize=(10, 8))
                plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                                      title=f'RAF-DB 混淆矩阵 (准确率: {acc * 100:.2f}%)')

                plt.savefig(
                    os.path.join('checkpoints', f"rafdb_epoch{epoch}_acc{acc:.4f}_bacc{balanced_acc:.4f}.png"))
                plt.close()


if __name__ == "__main__":
    run_training()