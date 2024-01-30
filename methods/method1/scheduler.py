import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupLinearLR(LRScheduler):
	def __init__(self,
	             optimizer,
	             warmup_steps,
	             total_steps,
	             min_proportion=0.0,
	             last_epoch=-1,
	             verbose=False):
		
		self.warmup_steps = warmup_steps
		self.max_steps = (total_steps - min_proportion * warmup_steps) / (1.0 - min_proportion)
		super(WarmupLinearLR, self).__init__(optimizer, last_epoch, verbose)
	
	def get_lr(self):
		if self.last_epoch == 0:
			return [group['lr'] * 0.1 / self.warmup_steps for group in self.optimizer.param_groups]
		
		if self.last_epoch > self.max_steps:
			return [group['lr'] for group in self.optimizer.param_groups]
		
		if self.last_epoch < self.warmup_steps:
			return [group['initial_lr'] * self.last_epoch / self.warmup_steps for group in self.optimizer.param_groups]
		else:
			return [group['initial_lr'] * (self.max_steps - self.last_epoch) / (self.max_steps - self.warmup_steps) for
			        group in self.optimizer.param_groups]
	
	def _get_closed_form_lr(self):
		if self.last_epoch < self.warmup_steps:
			return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
		else:
			return [base_lr * (self.max_steps - self.last_epoch) / (self.max_steps - self.warmup_steps) for base_lr
			        in self.base_lrs]


class WarmupLR(LRScheduler):
	def __init__(self,
	             optimizer,
	             warmup_steps,
	             min_proportion=0.0,
	             last_epoch=-1,
	             verbose=False):
		self.warmup_steps = warmup_steps
		self.min_proportion = min_proportion
		super(WarmupLR, self).__init__(optimizer, last_epoch, verbose)
	
	def get_lr(self):
		if self.last_epoch == 0:
			return [group['lr'] * 0.1 / self.warmup_steps for group in self.optimizer.param_groups]
		
		if self.last_epoch < self.warmup_steps:
			return [group['initial_lr'] * self.last_epoch / self.warmup_steps for group in self.optimizer.param_groups]
		else:
			reduce_steps = (self.last_epoch - self.warmup_steps) * 0.5 / self.warmup_steps
			coef = (self.min_proportion * reduce_steps + 1.0) / (reduce_steps + 1.0)
			return [group['initial_lr'] * coef for group in self.optimizer.param_groups]
	
	def _get_closed_form_lr(self):
		if self.last_epoch < self.warmup_steps:
			return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
		else:
			reduce_steps = (self.last_epoch - self.warmup_steps) * 0.5 / self.warmup_steps
			coef = (self.min_proportion * reduce_steps + 1.0) / (reduce_steps + 1.0)
			return [base_lr * coef for base_lr in self.base_lrs]
