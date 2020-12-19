# All standard variable types are supported.
SEED = 100


class OptimizerParams:
    def __init__(self):
        self.lr = 1e-3
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        self.weight_decay = 1e-2
        self.amsgrad = False


class ModelParams:
    def __init__(self):
        self.in_chan = 3
        self.num_neighborhoods = 16


class DataParams:
    def __init__(self):
        self.num_points = 4096
        self.batch_size = 32
        self.epochs = 100
