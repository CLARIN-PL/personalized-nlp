# TODO: We can isolate dedicated class for each type (random, fasttext, etc.)
# TODO: I don't think we have a time for that.

"""Single class responsible for creation of each possible embedding."""
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch

from settings import EMBEDDINGS_SIZES, TRANSFORMER_MODEL_STRINGS
from personalized_nlp.utils.embeddings import create_embeddings


class EmbeddingsCreator:
    """Create embeddings based on provided specification."""

    def __init__(
        self,
        directory: Path,
        embeddings_type: str,
        use_cuda: bool
    ) -> None:
        """Initialize object.

        Args:
            directory: The directory under which embeddings will be stored.
            embeddings_type: Type of embeddings.
            use_cuda: Whether to use CUDA during training.

        """
        if embeddings_type not in EMBEDDINGS_SIZES:
            raise Exception(f"Embedding type {embeddings_type} is invalid.")
        self.directory = directory
        # TODO: Not sure why but CUDA might not be available here
        # That was the same in original code
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.embeddings_type = embeddings_type
        self.personalised_embeddings_type = ""

    @property
    def embeddings_path(self):
        file_name = f"text_id_to_emb_{self.embeddings_type}.p"
        if self.personalised_embeddings_type != "":
            file_name = f"personalised_{self.personalised_embeddings_type}_{file_name}"
        return file_name

    @property
    def text_embedding_dim(self) -> int:
        """Get the dimension of text embeddings.

        Returns:
            The dimension of text embeddings.

        """
        if self.embeddings_type not in EMBEDDINGS_SIZES:
            raise NotImplementedError()

        return EMBEDDINGS_SIZES[self.embeddings_type]

    def set_personalised_embeddings_type(
        self,
        personalisation_type: str
    ):
        """Set value of `personalized_embeddings_type` field.

        Value indicates usage of personalized data in embeddings.
        Currently available options are:
            - "": no personalisation.
            - "annotator_id": add `annotator_id` to texts.
        Args:
            personalisation_type (str, optional): Personalisation type.
        """

        self.personalised_embeddings_type = personalisation_type

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get the texts embeddings.

        Returns:
            The text embeddings.

        """
        if not self.path.exists():
            self._create_embeddings(texts)
        return self._load_embeddings()

    def _load_embeddings(self) -> torch.Tensor:
        with open(self.path, "rb") as f:
            text_idx_to_emb = pickle.load(f)

        embeddings = []
        for text_id in range(len(text_idx_to_emb.keys())):
            embeddings.append(text_idx_to_emb[text_id])

        embeddings = np.array(embeddings)
        return torch.tensor(embeddings)

    def _create_embeddings(self, texts: List[str]):
        """Create embeddings under provided path if not already available."""

        if self.embeddings_type in TRANSFORMER_MODEL_STRINGS:
            model_name = TRANSFORMER_MODEL_STRINGS[self.embeddings_type]
        else:
            model_name = self.embeddings_type

        create_embeddings(
            texts,
            self.embeddings_path,
            model_name=model_name,
            use_cuda=self.use_cuda,
        )
