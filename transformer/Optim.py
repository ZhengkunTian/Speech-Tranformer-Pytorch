'''A wrapper class for optimizer '''
import numpy as np


class ScheduledOptim(object):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, multi_gpu=False):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.current_step = 0
        self.lr = 0
        self.multi_gpu = multi_gpu

    def step(self):
        "Step by the inner optimizer"
        self.update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.current_step += 1
        lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.current_step, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.current_step])

        if not self.multi_gpu:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                self.lr = lr
        else:
            for param_group in self.optimizer.module.param_groups:
                param_group['lr'] = lr
                self.lr = lr
