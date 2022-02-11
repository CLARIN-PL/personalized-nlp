from pytorch_lightning.callbacks import Callback
import torch
import numpy as np


class SaveOutputsCallback(Callback):

    def __init__(self, save_dir=None) -> None:
        super().__init__()
        self.test_outputs = []
        self.test_ys = []
        self.save_dir = save_dir

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        print(outputs)
        raise None
        self.test_outputs.append(outputs['output'])
        self.test_ys.append(outputs['y'])

    def on_test_end(self, *args, **kwargs):
        if not self.test_outputs:
            return

        test_outputs = torch.cat(self.test_outputs)
        test_ys = torch.cat(self.test_ys)

        if test_outputs.shape[1] != test_ys.shape[1]:
            test_outputs = torch.argmax(test_outputs, dim=1)
            test_ys = test_ys.flatten()
        
        test_outputs = test_outputs.cpu().numpy()
        test_ys = test_ys.cpu().numpy()

        self.final_test_outputs = test_outputs
        self.final_test_ys = test_ys

        if self.save_dir:
            with open(f'{self.save_dir}/outputs', 'wb') as f:
                np.save(f, self.final_test_outputs)
            with open(f'{self.save_dir}/ys', 'wb') as f:
                np.save(f, self.final_test_ys)
