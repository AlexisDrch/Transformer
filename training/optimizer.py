import torch


class NoamOpt(object):
    """
    The authors specify that they used the Adam optimizer with:

    .. math::

        \\beta_1 = 0.9, \\beta_2 = 0.98, \\epsilon = 10^{−9}

    But varied the learning rate over the course of training, according to the formula:

    .. math::

        l_{rate} = {d_{model}}^{-0.5} \\cdot min({step\_num}^{−0.5}, step_num \\cdot {warmup\_steps}^{−1.5})


    This corresponds to increasing the learning rate linearly for the first `warmup_steps` training steps,
    and decreasing it thereafter proportionally to the inverse square root of the step number.
    """

    def __init__(self, model: torch.nn.Module, model_size=512, lr=0., betas=(0.9, 0.98), eps=1e-9, factor=2, warmup=4000):
        """
        Constructor for the specific Optimizer used for training.

        :param model: Model to optimize.

        :param model_size: Overall vector dimension used in model. Default: 512.

        :param lr: learning rate.

        :param betas: Coefficients used for computing running averages of gradient and its square

        :param eps: Term added to the denominator to improve numerical stability.

        :param factor: Multiplicative factor for the learning rate. Default: 2.

        :param warmup: Number of warmup steps. Default: 4000.
        """

        self.optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                          betas=betas, eps=eps)
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self) -> None:
        """
        Performs an optimization step on the model's parameters.
        Also update the learning rate based on the above formula.

        """
        # increment step index
        self._step += 1

        # update learning rate
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

        # perform optimization step
        self.optimizer.step()

    def rate(self) -> float:
        """
        Compute updated learning rate based on step index and formula.

        """
        return self.factor * (self.model_size ** (-0.5) *
                              min(self._step ** (-0.5), self._step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
