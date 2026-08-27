import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                # Linear warmup
                lr = base_lr * float(step) / float(max(1, self.warmup_steps))
            elif step > self.total_steps:
                # Decay finish limit
                lr = self.min_lr
            else:
                # Cosine decay
                progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + coeff * (base_lr - self.min_lr)
            lrs.append(lr)
        return lrs
