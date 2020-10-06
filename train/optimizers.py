import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class Optimizer(object):
    def __init__(self, method, learning_rate, max_grad_norm,
                 lr_decay=1, start_decay_steps=None, decay_steps=None,
                 beta1=0.9, beta2=0.999,
                 adagrad_accum=0.0,
                 decay_method=None,
                 warmup_steps=4000,
                 model_size=None,
                 warmup_start_lr = 0,optims="fairseq"):
        self.last_ppl = None
        self.learning_rate = learning_rate
        self.original_lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_steps = start_decay_steps
        self.decay_steps = decay_steps
        self._step = 1
        self.betas = [beta1, beta2]
        self.adagrad_accum = adagrad_accum
        self.decay_method = decay_method
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        self.types=optims
        print("Learning Rate:" + self.types)
        self.lr_step = (learning_rate - warmup_start_lr) / self.warmup_steps
        self.decay_factor = learning_rate * self.warmup_steps ** 0.5
        self.cus_lr = warmup_start_lr
        self.warm_start_lr = warmup_start_lr
        if optims == "fairseq":
            self.learning_rate = warmup_start_lr
        #self.setlrs(warmup_start_lr)


    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_parameters(self, model):
        """ ? """
        params = [p for p in model.parameters() if p.requires_grad]
        if self.method == 'sgd':
            self.optimizer = optim.SGD(params, lr=self.learning_rate)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(
                params,
                lr=self.learning_rate,
                initial_accumulator_value=self.adagrad_accum)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(params, lr=self.learning_rate)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(params, lr=self.learning_rate,
                                        betas=self.betas, eps=1e-9, weight_decay=0,amsgrad=False)
        elif self.method == 'adamax':
            self.optimizer = optim.Adamax(params,lr=self.learning_rate,betas=self.betas,eps=1e-9,weight_decay=0)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)
        self.setlrs(self.learning_rate)


    def setlrs(self,lr):

        self.learning_rate =  lr
        for group in self.optimizer.param_groups:
            if self.method != 'adafactor':
                group['lr'] = self.learning_rate
            if self.max_grad_norm:
                clip_grad_norm_(group['params'], self.max_grad_norm)
    def reloadlrs(self,learning_rate):
        self.decay_factor = learning_rate * self.warmup_steps ** 0.5
        self.lr_step = (learning_rate - self.warm_start_lr) / self.warmup_steps
    def step(self):
        if self.types != "fairseq":
            self.step2()
            return

        self._step += 1

        if self._step < self.warmup_steps:
            self.learning_rate = self.warm_start_lr + self._step * self.lr_step
        else:
            self.learning_rate = self.decay_factor * self._step**-0.5

        self.setlrs(self.learning_rate)
        self.optimizer.step()

    def step2(self):

        self._step += 1

        if self.decay_method == "noam":
            lr_scale = (
                self.model_size ** (-0.5) *
                min(self._step ** (-0.5),
                    self._step * self.warmup_steps**(-1.5)))
        elif self.start_decay_steps is not None:
            step = self._step - self.start_decay_steps
            lr_scale = (self.lr_decay ** (
                max(step + self.decay_steps, 0) // self.decay_steps))
        else:
            lr_scale = 1

        self.learning_rate = lr_scale * self.original_lr
        for group in self.optimizer.param_groups:
            if self.method != 'adafactor':
                group['lr'] = self.learning_rate
            if self.max_grad_norm:
                clip_grad_norm_(group['params'], self.max_grad_norm)
        self.optimizer.step()
