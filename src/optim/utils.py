import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR
import warnings
from typing import Tuple
from transformers import get_cosine_schedule_with_warmup
from .muon import Muon
from .muonq import MuonQ


def get_optimizer_factory(name: str):
    """
    Returns the optimizer class corresponding to the given name.
    """
    name = name.lower()
    if name == 'sgd':
        return torch.optim.SGD
    elif name == 'adam':
        return torch.optim.Adam
    elif name == 'adamw':
        return torch.optim.AdamW
    elif name == 'muon':
        return Muon
    elif name == 'muonq':
        return MuonQ
    else:
        raise ValueError(f"Unknown optimizer name: {name}")

def get_scheduler(config: dict, opt: torch.optim.Optimizer, total_iterations = None) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Main function mapping to a learning rate scheduler.
    """
    # if not specified, use constant step sizes
    name = config.get('name', 'constant')
    
    if name == 'constant':
        lr_fun = lambda epoch: 1 # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
    
    elif name == 'linear':
        lr_fun = lambda epoch: 1/(epoch+1) # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    elif name == 'sqrt':
        lr_fun = lambda epoch: (epoch+1)**(-1/2) # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    elif 'exponential' in name:
        # use sth like 'exponential_60_0.5': decay by factor 0.5 every 60 epochs
        step_size = int(name.split('_')[1])
        gamma = float(name.split('_')[2])
        scheduler = StepLR(opt, step_size=step_size, gamma=gamma)

    elif 'warm-up-cosine' in name:
        num_warmup_steps = int(config['warm_up_fraction'] * total_iterations) 
        scheduler = get_cosine_schedule_with_warmup(
                    opt,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_iterations
                    )
    elif 'constant-linear' in name:  # New scheduler
        num_warmup_steps = int(config['warm_up_fraction'] * total_iterations)

        # def get_lr(step: int):
        #     x = step / total_iterations  # progress in training
        #     assert 0 <= x < 1
        #     if x < config['warm_up_fraction']:
        #         return 1.0  # Stable learning rate
        #     else:
        #         w = (1 - x) / (1-config['warm_up_fraction'])
        #         return w * 1.0 + (1 - w) * 0.1  # Linear decay during cooldown


        def get_lr(step):
            if step < num_warmup_steps:
                return 1.0  # Constant learning rate during warm-up
            else:
                # Linearly decay after warm-up
                return max(0.1, 1.0 - (step - num_warmup_steps) / (total_iterations - num_warmup_steps))

        scheduler = LambdaLR(opt, lr_lambda=get_lr)

        
    else:
        raise ValueError(f"Unknown learning rate schedule name {name}.")
    
    return scheduler
