from collections import defaultdict

import pandas as pd
import torch
from pytorch_lightning.callbacks import Callback


class SaveConfidencesCallback(Callback):
    def __init__(self, save_dir=None, data_columns=None) -> None:
        super().__init__()
        self._outputs: list[torch.Tensor] = []

    def on_predict_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self._outputs.append(outputs["output"])

    def on_predict_end(self, *args, **kwargs):
        if not self._outputs:
            return

        outputs = torch.cat(self._outputs)

        self.predict_outputs = outputs.cpu().numpy()
