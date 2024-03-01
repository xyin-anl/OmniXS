from DigitalBeamline.digitalbeamline.extern.m3gnet.featurizer import (
    _load_default_featurizer,
)
from matgl.models._m3gnet import M3GNet
from torch import nn


class XASM3GNet(M3GNet):
    def __init__(
        self,
        trained=True,
        ntargets=141,
        *args,
        **kwargs,
    ):
        super().__init__(ntargets=ntargets, *args, **kwargs)
        if trained:
            self._load()

    def _load(self):
        model = _load_default_featurizer()
        self.load_state_dict(
            model.state_dict(),
            strict=False,  # coz the last layer is different
        )
        return self


if __name__ == "__main__":
    model = XASM3GNet()
    print(model.final_layer)
