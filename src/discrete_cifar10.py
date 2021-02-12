from __future__ import print_function
import csv
from itertools import zip_longest
import torch.nn as nn
from layers import DiscretizedLinear, Discretization, DiscretizedConv2d
from training_routines import train, test
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR  # manager of lr decay

class Cifar10Conv3NN(nn.Module):
    def __init__(self, discrt_lvls=2.):
        super(Cifar10Conv3NN, self).__init__()
        self.pool = nn.MaxPool2d(2)
        
        self.dp = 0.2
        self.discrt_lvls = discrt_lvls

        self.layer128_1 = nn.Sequential(
            DiscretizedConv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, discrt_lvls=discrt_lvls),
            nn.BatchNorm2d(128),
            nn.Dropout2d(self.dp),
            nn.CELU()
        )

        self.layer128_2 = nn.Sequential(
            DiscretizedConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, discrt_lvls=discrt_lvls),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(self.dp),
            nn.CELU()
        )

        self.layer256_1 = nn.Sequential(
            DiscretizedConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, discrt_lvls=discrt_lvls),
            nn.BatchNorm2d(256),
            nn.Dropout2d(self.dp),
            nn.CELU()
        )

        self.layer256_2 = nn.Sequential(
            DiscretizedConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, discrt_lvls=discrt_lvls),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(self.dp),
            nn.CELU()
        )

        self.layer512_1 = nn.Sequential(
            DiscretizedConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, discrt_lvls=discrt_lvls),
            nn.BatchNorm2d(512),
            nn.Dropout2d(self.dp),
            nn.CELU()
        )

        self.layer512_2 = nn.Sequential(
            DiscretizedConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, discrt_lvls=discrt_lvls),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Dropout2d(self.dp),
            nn.CELU()
        )

        self.fc_layer1 = nn.Sequential(
            DiscretizedLinear(in_features=512 * 4 * 4, out_features=1024, discrt_lvls=discrt_lvls),
            nn.BatchNorm1d(1024),
            nn.Dropout(self.dp),
            nn.CELU()
        )

        self.fc_layer2 = nn.Sequential(
            DiscretizedLinear(in_features=1024, out_features=1024, discrt_lvls=discrt_lvls),
            nn.BatchNorm1d(1024),
            nn.Dropout(self.dp),
            nn.CELU()
        )

        self.fc_layer3 = nn.Sequential(
            DiscretizedLinear(in_features=1024, out_features=10, discrt_lvls=discrt_lvls),
            nn.BatchNorm1d(10),
        )

    def forward(self, x):
        x = self.layer128_1(x)
        x = self.layer128_2(x)
        x = self.layer256_1(x)
        x = self.layer256_2(x)
        x = self.layer512_1(x)
        x = self.layer512_2(x)
        x = x.view(-1, 4*4*512)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        return x

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 BNN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda-num', type=int, default=0,
                        help='Choses GPU number')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--weight-decay', type=float, default=0, metavar='W',
                        help='coefficient of L2 regulariztion')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--discretization', type=float, default=1., metavar='M',
                        help='Discretization level number per length 1 (default 1, means binary)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    DISCRETE_LVLS_NUMBER = 8

    torch.manual_seed(args.seed)

    device = torch.device("cuda:%d" % args.cuda_num if torch.cuda.is_available() else "cpu")
    print("Use device:", device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomAffine(degrees=35, shear=0.2),
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Cifar10Conv3NN(discrt_lvls=DISCRETE_LVLS_NUMBER).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[20,80,150,250,400], gamma=0.7)
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  # managinng lr decay

    test_accuracy = []
    train_accuracy = []

    for epoch in range(1, args.epochs + 1):
        print('Epoch:', epoch, 'LR:', scheduler.get_lr())
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, train_loader, test_accuracy, train_accuracy)
        scheduler.step(epoch=epoch)
        if epoch > 10:
            if (args.save_model):
                torch.save(model.state_dict(), f"../model/cifar10_conv_discrt_lvls_{DISCRETE_LVLS_NUMBER}.pt")

            d = [train_accuracy, test_accuracy]
            export_data = zip_longest(*d, fillvalue='')
            with open(f'../model/cifar10_conv_discrt_lvls_{DISCRETE_LVLS_NUMBER}_report.csv', 'w', encoding="ISO-8859-1", newline='') as report_file:
                wr = csv.writer(report_file)
                wr.writerow(("Train accuracy", "Test accuracy"))
                wr.writerows(export_data)
            report_file.close()

if __name__ == '__main__':
    main()
