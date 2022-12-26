# from _base import TrainerBase
import os
import torch
import numpy as np
import math

from torch.optim.lr_scheduler import _LRScheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

from dataset import dataset, sampler
from model import network


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
            self,
            optimizer,
            successor,
            warmup_epoch,
            last_epoch=-1,
            verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
            self,
            optimizer,
            successor,
            warmup_epoch,
            last_epoch=-1,
            verbose=False
    ):
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        else:
            return [
                lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
            ]


def linear_warmup_cosine_decay(epoch):
    # linear warmup
    epoch_warmup = 10
    total_epoch = 100
    if epoch < epoch_warmup:
        return 0.9 * epoch / epoch_warmup + 0.1
    # cosine decay
    decay_lr = 0.5 * (1 + math.cos(math.pi * (epoch - epoch_warmup) / (total_epoch - epoch_warmup)))
    return max(decay_lr, 0.1)


class Trainer:
    def __init__(self, args):
        super(Trainer, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._config = args
        self.build_data_loder()
        self.build_model()

        self.image_criterion = torch.nn.CrossEntropyLoss()
        self.mouth_criterion = torch.nn.CrossEntropyLoss()

        self.best_epoch = 0
        self.best_auc = 0.0
        self.best_model_wts = 0.0

        self.outdir = self._config.save_dir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)

    def build_data_loder(self):
        manipulation_method = self._config.fake_type
        batch_size = self._config.BATCH_SIZE
        num_workers = self._config.NUM_WORKERS

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.train_dataset = dataset.ForensicsClips(mode='train', frames_per_clip=self._config.clip_size,
                                                    fakes=manipulation_method, transform=train_transform, )
        self.val_dataset = dataset.ForensicsClips(mode='val', frames_per_clip=self._config.clip_size,
                                                  fakes=manipulation_method, transform=val_transform, )

        # self.train_sampler = sampler.ConsecutiveClipSampler(self.train_dataset.clips_per_video)
        # self.val_sampler = sampler.ConsecutiveClipSampler(self.val_dataset.clips_per_video)

        self.train_sampler = sampler.RandomClipSampler(self.train_dataset.clips_per_video)
        self.val_sampler = sampler.RandomClipSampler(self.val_dataset.clips_per_video)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, sampler=self.train_sampler,
                                           num_workers=num_workers)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, sampler=self.val_sampler)

    def build_model(self):

        self.model = network.TransFuse(self._config.crops_pretrained_model, self._config. mouth_pretrained_model)
        self.model.to(self.device)

        self.optimizer = self.build_optimizer()
        self.lr_scheduler = self.build_lr_scheduler()

        device_count = torch.cuda.device_count()

        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = torch.nn.DataParallel(self.model)

    def build_optimizer(self):
        optim = self._config.OPTIMIZING_METHOD
        lr = np.float32(self._config.BASE_LR)
        weight_decay = np.float32(self._config.WEIGHT_DECAY)
        momentum = np.float32(self._config.MOMENTUM)

        if isinstance(self.model, torch.nn.Module):
            param_groups = self.model.parameters()
        else:
            param_groups = self.model

        if optim == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optim == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise NotImplementedError(f"Optimizer {optim} not implemented yet!")

        return optimizer

    def build_lr_scheduler(self):
        lr_policy = self._config.LR_POLICY
        max_epoch = self._config.MAX_EPOCH
        warmup_epoches = self._config.WARMUP_EPOCHS

        if lr_policy == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, float(max_epoch))

        if warmup_epoches > 0:
            scheduler = LinearWarmupScheduler(
                self.optimizer, scheduler, warmup_epoches,
                last_epoch=0
            )

        return scheduler

    def before_train(self):
        log_dir = self._config.log_dir
        writer_dir = os.path.join(log_dir, 'tensorboard')

        if not os.path.exists(writer_dir):
            try:
                os.makedirs(writer_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

    def run(self):
        self.before_train()
        for epoch in range(self._config.MAX_EPOCH):
            self.train_per_epoch(epoch)
            self.val_per_epoch(epoch)

        self.model.load_state_dict(self.best_model_wts)
        torch.save(self.model.state_dict(), os.path.join(self.outdir, 'best_model_{}.pth'.format(self.best_epoch)))

    def train_per_epoch(self, epoch):
        # training mode
        self.model.train()

        for batch_id, data in enumerate(self.train_dataloader):

            num_iters = epoch * len(self.train_dataloader) + batch_id

            clips_img, clips_mouth, rois, labels, video_indices = data
            clips_img, clips_mouth, labels = clips_img.to(self.device), clips_mouth.to(self.device), labels.to(self.device).unsqueeze(1).to(
                torch.float32)

            _, _, image_outputs, mouth_outputs = self.model(clips_img, clips_mouth, rois)
            loss = (1 - self._config.BETA) * self.image_criterion(image_outputs, labels) + self._config.BETA * self.mouth_criterion(mouth_outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metrics = self.compute_metrics(loss)

            loss_data = metrics['loss_data']
            lr = metrics['lr']

            # logger
            if batch_id % 50 == 0:
                print(f"epoch [{epoch + 1}] / batch_id [{batch_id}] loss: {loss_data:.4f} lr: {lr}")

            self._writer.add_scalar('train/loss', loss_data, num_iters)
            self._writer.add_scalar('train/lr', lr, num_iters)

    def val_per_epoch(self, epoch):
        self.model.eval()

        video_to_logits = defaultdict(list)
        video_to_labels = {}
        epoch_loss = 0
        epoch_corrects = 0

        with torch.no_grad():
            for data in tqdm(self.val_dataloader):
                clips_img, clips_mouth, rois, labels, video_indices = data
                clips_img, clips_mouth, labels = clips_img.to(self.device).to(torch.float32), clips_mouth.to(self.device).to(torch.float32), labels.to(self.device).unsqueeze(1).to(
                    torch.float32)
                _, _, image_outputs, mouth_outputs = self.model(clips_img, clips_mouth, rois)
                loss = (1 - self._config.BETA) * self.image_criterion(image_outputs,labels) + self._config.BETA * self.mouth_criterion(mouth_outputs, labels)

                loss_data = loss.data.item()
                epoch_loss += loss_data

                for i in range(len(video_indices)):
                    video_id = video_indices[i].item()
                    video_to_logits[video_id].append(logits[i])
                    video_to_labels[video_id] = labels[i]

            epoch_loss = epoch_loss / len(self.val_dataset)
            epoch_auc = self.compute_video_level_auc(video_to_logits, video_to_labels)

            print(f"epoch [{epoch + 1}] val loss:{epoch_loss: .4f} val auc:{epoch_auc:.4f}")

            if epoch_auc > self.best_auc:
                self.best_auc = epoch_auc
                self.best_epoch = epoch + 1
                self.best_model_wts = self.model.state_dict()

        self.lr_scheduler.step()
        torch.save(self.model.state_dict(), os.path.join(self.outdir, 'model_epoch{}.pth'.format(epoch + 1)))

        self._writer.add_scalar("val/auc", epoch_auc, epoch)

    def compute_metrics(self, loss):
        metrics = {}
        loss = loss.data.item()
        metrics['loss_data'] = loss / self._config.BATCH_SIZE
        metrics['lr'] = self.optimizer.param_groups[0]['lr']

        return metrics

    def compute_video_level_auc(self, video_to_logits, video_to_labels):
        """ "
            Compute video-level area under ROC curve. Averages the logits across the video for non-overlapping clips.

            Parameters
            ----------
            video_to_logits : dict
                Maps video ids to list of logit values
            video_to_labels : dict
                Maps video ids to label
            """
        output_batch = torch.stack(
            [torch.mean(torch.stack(video_to_logits[video_id]), 0, keepdim=False) for video_id in
             video_to_logits.keys()]
        )
        output_labels = torch.stack([video_to_labels[video_id] for video_id in video_to_logits.keys()])

        fpr, tpr, _ = metrics.roc_curve(output_labels.cpu().numpy(), output_batch.cpu().numpy())
        return metrics.auc(fpr, tpr)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='train CDIN with FaceForensics++ for deepfake detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--save_dir', type=str, default='',
                        help='the save path of the output models')
    parser.add_argument('--log_dir', type=str, default='',
                        help='the save path of the log')
    parser.add_argument('--crops_pretrained_model', type=str, default='',
                        help='the pre-trained model for GAD-branch')
    parser.add_argument('--mouth_pretrained_model', type=str, default='',
                        help='the pre-trained model for LRM-branch')
    parser.add_argument('--fake_type', type=str, default='',
                        choices=['youtube', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
                        help='which subset of the dataset')
    parser.add_argument('--clip_size', type=int, default=16)
    parser.add_argument('--MAX_EPOCH', type=int, default=30)
    parser.add_argument('--BATCH_SIZE', type=int, default=2)
    parser.add_argument('--LR_POLICY', type=str, )
    parser.add_argument('--WARMUP_EPOCHS', type=int, )
    parser.add_argument('--OPTIMIZING_METHOD', type=str, default='adam')
    parser.add_argument('--BASE_LR', type=int, default=1e-5)
    parser.add_argument('--WEIGHT_DECAY', type=int, default=1e-6)
    parser.add_argument('--MOMENTUM', type=int,)
    parser.add_argument('--NUM_WORKERS', type=int, default=2)
    parser.add_argument('--BETA', type=int, default=0.5)

    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.run()