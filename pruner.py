import torch
import torch.nn as nn

__all__ = ['snip', 'prue']


def snip(model, loss, dataloader, sparsity, gpu):
    '''
    SNIP: SINGLE-SHOT NETWORK PRUNING BASED ON CONNECTION SENSITIVITY
    https://openreview.net/forum?id=B1VZqjAcYX
    '''
    model.train()
    device = torch.device(f"cuda:{gpu}")
    model.to(device)
    total = 0

    for name, buf in model.named_buffers():
        if name.endswith('_mask'):
            buf.requires_grad = True
            total += buf.numel()

    model.zero_grad()
    # inputs, targets = next(iter(dataloader))
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model.forward(inputs)
        loss(outputs, targets).backward()

    index = 0
    global_score = torch.zeros(total)
    for name, buf in model.named_buffers():
        if name.endswith('_mask'):
            size = buf.numel()
            global_score[index:(index+size)] = buf.grad.view(-1).abs().clone()
            index += size

    kth = sparsity * total
    thre, _ = torch.kthvalue(global_score, int(kth))

    with torch.no_grad():
        for name, buf in model.named_buffers():
            if name.endswith('_mask'):
                buf.requires_grad = False
                mask = buf.grad.abs().gt(thre).float()
                buf.mul_(mask)

    model.zero_grad()
    model.to('cpu')


def prue(model, loss, dataloader, sparsity, gpu):
    model.eval()
    device = torch.device(f"cuda:{gpu}")

    model.to(device)
    total = 0

    # Make the pruning mask differentiable
    for name, buf in model.named_buffers():
        if name.endswith('_mask'):
            buf.requires_grad = True
            total += buf.numel()

    num_classes = max(dataloader.dataset.targets) + 1

    # The trick mentioned in Section 4. First calculate the average prediction ~f(x)[c] for each class
    with torch.no_grad():
        predict = [[] for _ in range(num_classes)]
        for img, target in dataloader:
            img, target = img.to(device), target.to(device)
            output = model(img)
            softmax_p = nn.functional.softmax(output, dim=1)
            for t, index in zip(softmax_p, target):
                predict[index].append(t[index])
                # predict[index] = torch.cat(
                #     (predict[index], t[index].view(-1))
                # )

        for i, t in enumerate(predict):
            predict[i] = torch.mean(torch.Tensor(t))

    # Only one label in each minibatch
    for img, target in dataloader:
        img, target = img.to(device), target.to(device)
        output = model(img)
        softmax_p = nn.functional.softmax(output, dim=1)

        all_std = 0
        for t, index in zip(softmax_p, target):
            all_std += torch.norm(predict[index] - t[index])
        all_std.backward()

    index = 0
    global_score = torch.zeros(total)
    for name, buf in model.named_buffers():
        if name.endswith('_mask'):
            size = buf.numel()
            global_score[index:(index+size)] = buf.grad.view(-1).abs().clone()
            index += size

    kth = sparsity * total
    thre, _ = torch.kthvalue(global_score, int(kth))

    # Apply the new pruning mask and turn off grad
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if name.endswith('_mask'):
                buf.requires_grad = False
                mask = buf.grad.abs().gt(thre).float()
                buf.mul_(mask)

    model.zero_grad()
    model.to('cpu')
