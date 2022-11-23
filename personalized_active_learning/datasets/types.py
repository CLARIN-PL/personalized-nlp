from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass()
class TextFeaturesBatch:
    """Defines contract between datasets & models. TextFeaturesBatch is models input."""

    # TODO: Consider change in approach - models operates on tensors.
    # TODO: Something else prepares data for them.
    text_ids: torch.Tensor
    raw_texts: np.ndarray
    embeddings: torch.Tensor
    annotator_ids: torch.Tensor
    annotator_biases: torch.Tensor


class TextFeaturesBatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        y: np.ndarray,
        text_ids: np.ndarray,
        raw_texts: np.ndarray,
        embeddings: np.ndarray,
        annotator_ids: np.ndarray,
        annotator_biases: np.ndarray,
    ) -> None:

        self.y = y
        self.text_ids = text_ids
        self.raw_texts = raw_texts
        self.embeddings = embeddings
        self.annotator_ids = annotator_ids
        self.annotator_biases = annotator_biases

    def __getitem__(self, index: np.ndarray) -> Tuple[TextFeaturesBatch, torch.Tensor]:
        """Obtain batch for texts of provided indices."""
        batch_y = self.y[index]
        batch_text_ids = self.text_ids[index]
        batch_annotator_ids = self.annotator_ids[index]

        text_features_batch = TextFeaturesBatch(
            text_ids=torch.tensor(batch_text_ids),
            annotator_ids=torch.tensor(batch_annotator_ids),
            raw_texts=self.raw_texts[batch_text_ids],
            embeddings=torch.tensor(self.embeddings[batch_text_ids]),
            annotator_biases=torch.tensor(self.annotator_biases[batch_annotator_ids]),
        )
        return text_features_batch, batch_y

    def __len__(self) -> int:
        return len(self.y)
