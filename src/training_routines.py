import torch
import torch.nn as nn
import torch.nn.functional as F

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # target_new = target.reshape(1, 1)
        # num_classes = 10
        # one_hot_target = (target_new == torch.arange(num_classes).reshape(1, num_classes).to(device)).float().add(-0.1).div(0.3)
        # loss = F.mse_loss(output, one_hot_target, reduction='sum')
        loss = F.cross_entropy(output, target, reduction='mean')
        loss.backward()
        # for name, param in model.named_parameters():
        #     print(f'{name} before step', f'mean val {param.abs().mean().item()}', f'mean grad: {param.grad.abs().mean().item()}', f'max val {param.max().item()}', f'max grad: {param.grad.max().item()}', f'min val {param.min().item()}', f'min grad: {param.grad.min().item()}')
        optimizer.step()
        # for name, param in model.named_parameters():
        #     print(f'{name} after step', f'mean val {param.abs().mean().item()}', f'mean grad: {param.grad.abs().mean().item()}', f'max val {param.max().item()}', f'max grad: {param.grad.max().item()}', f'min val {param.min().item()}', f'min grad: {param.grad.min().item()}')
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, train_loader=None, test_accuracy=None, train_accuracy=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # target_new = target.reshape(1, 1)
            # num_classes = 10
            # one_hot_target = (target_new == torch.arange(num_classes).reshape(1, num_classes).to(device)).float().add(-0.1).div(0.3)
            # test_loss += F.mse_loss(output, one_hot_target, reduction='sum').item()
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if test_accuracy is not None:
        test_accuracy.append(100. * correct / len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if not train_loader is None:
        correct = 0
        train_loss = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # target_new = target.reshape(1, 1)
                # num_classes = 10
                # one_hot_target = (target_new == torch.arange(num_classes).reshape(1, num_classes).to(device)).float().add(-0.1).div(0.3)
                # test_loss += F.mse_loss(output, one_hot_target, reduction='sum').item()
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss /= len(train_loader.dataset)

        if train_accuracy is not None:
            train_accuracy.append(100. * correct / len(train_loader.dataset))

        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
