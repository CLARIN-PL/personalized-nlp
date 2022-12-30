import pandas as pd

from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple


class PersonalisedEmbeddings(ABC):
    """The base class from each personalised embeddings class should derive."""

    def __init__(
        self,
        texts_data: pd.DataFrame,
        annotations: pd.DataFrame
    ) -> None:
        """Initialize object.

        Args:
            texts_data (pd.DataFrame): Dataset with text data.
            annotations (pd.DataFrame): Dataset with annotations data.
        """
        super().__init__()

        self.data = texts_data
        self.annotations = annotations

    @abstractproperty
    def name(self) -> str:
        """The name of used personalisation.

        Property used as filename in embeddings creator.

        Returns:
            str: Personalisation name
        """
        raise NotImplementedError()

    @abstractmethod
    def apply_personalisation(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply defined personalistation.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Modified datasets -
                `annotations` and `text_data`
        """
        raise NotImplementedError()
