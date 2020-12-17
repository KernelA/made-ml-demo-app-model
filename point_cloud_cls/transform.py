from torch_geometric import transforms as trans


class BaseTransform:
    def __init__(self, num_points: int) -> None:
        super().__init__()
        self.num_points = num_points
        self._tranforms = trans.Compose([trans.SamplePoints(
            num=num_points), trans.Center(), trans.NormalizeScale()])

    def __call__(self, data):
        return self._tranforms(data)


class TrainTransform(BaseTransform):
    def __init__(self, num_points: int, angle_degree: float, axis: int, rnd_shift: float) -> None:
        assert axis in (0, 1, 2)
        super().__init__(num_points=num_points)
        self._augm = trans.Compose([trans.RandomRotate(angle_degree, axis=axis), trans.RandomTranslate(rnd_shift)])

    def __call__(self, data):
        transformed = super().__call__(data)
        return self._augm(transformed)
