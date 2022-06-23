import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class TrainingModule(pl.LightningModule):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model

        self.save_hyperparameters(
            'lr', 'weight_decay', 'milestones', 'gamma', 'arch', 'batch_size', 'dataset')

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        self.accuracy(output, target)
        return output

    def validation_epoch_end(self, output):
        self.log('valid_acc', self.accuracy.compute(), prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.accuracy.reset()
        self.val_loss, self.val_count = 0, 0

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.hparams.milestones, self.hparams.gamma)
        return [optimizer], [scheduler]


class SoftTarget(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''

    def __init__(self, T=4.0):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss


class DistillModule(TrainingModule):
    def __init__(self, model, t_model, **kwargs):
        super().__init__(model, **kwargs)
        self.t_model = t_model
        self.criterionKD = SoftTarget(kwargs['temperature'])
        self.alpha = 0.9
        self.t_model.eval()

    def training_step(self, batch, batch_idx):
        data, target = batch

        with torch.no_grad():
            t_output = self.t_model(data)
        output = self(data)
        cls_loss = F.cross_entropy(output, target)
        kd_loss = self.criterionKD(output, t_output.detach())
        return (1 - self.alpha) * kd_loss + self.alpha * cls_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smooth=0.2):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smooth = smooth

    def forward(self, x, target):
        confidence = 1. - self.smooth
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smooth * smooth_loss
        return loss.mean()


class SmoothModule(TrainingModule):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.criterionKD = LabelSmoothingCrossEntropy()

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        kd_loss = self.criterionKD(output, target)
        return kd_loss
