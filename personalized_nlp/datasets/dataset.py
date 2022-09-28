from typing import Dict, Optional, Tuple

import numpy as np
import torch.utils.data


# TODO specify types!
# TODO add docstring!
class BatchIndexedDataset(torch.utils.data.Dataset):

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 text_features: Optional[Dict] = None,
                 annotator_features: Optional[Dict] = None) -> None:

        self.X = X
        self.y = y

        if text_features is not None:
            self.text_features = text_features
        else:
            self.text_features = {}

        if annotator_features is not None:
            self.annotator_features = annotator_features
        else:
            self.annotator_features = {}

    def __getitem__(
            self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        text_ids = self.X[index, 0]
        annotator_ids = self.X[index, 1]
        annotation_ids = self.X[index, 2]

        batch_data = {}

        batch_data['text_ids'] = text_ids
        batch_data['annotator_ids'] = annotator_ids
        batch_data["annotation_ids"] = annotation_ids

        for k, tf in self.text_features.items():
            batch_data[k] = tf[text_ids]

        for k, af in self.annotator_features.items():
            batch_data[k] = af[annotator_ids]

        batch_y = self.y[index]

        return batch_data, batch_y

    def __len__(self) -> int:
        return len(self.y)
