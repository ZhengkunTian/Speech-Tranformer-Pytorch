'''A wrapper class for optimizer '''
import numpy as np
import torch.optim as optim


class ScheduledOptim(object):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, model, d_model, config):
        self.lr = 0
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        self.d_model = d_model
        self.n_warmup_steps = config.n_warmup_steps
        # self.current_step = 0

    def step(self, global_step):
        "Step by the inner optimizer"
        self.update_learning_rate(global_step)
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, optimizer_state_dict):
        return self.optimizer.load_state_dict(optimizer_state_dict)

    def update_learning_rate(self, global_step):
        ''' Learning rate scheduling per step '''

        lr = np.power(self.d_model, -0.5) * np.min([
            np.power(global_step, -0.5),
            np.power(self.n_warmup_steps, -1.5) * global_step])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            self.lr = lr
