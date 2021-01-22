# All standard variable types are supported.
SEED = 100


class OptimizerParams:
    def __init__(self):
        self.lr = 1e-1
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        self.weight_decay = 1e-2
        self.amsgrad = False


class ModelParams:
    def __init__(self):
        self.in_chan = 3
        self.num_neighborhoods = 32


class DataParams:
    def __init__(self):
        self.num_points = 512


class TrainParams:
    def __init__(self) -> None:
        self.batch_size = 110
        self.epochs = 1
        self.valid_every = 5
        self.benchmark = True
        self.deterministic = True


class SchedulerParams:
    def __init__(self):
        self.step_size = 20
        self.verbose = True
