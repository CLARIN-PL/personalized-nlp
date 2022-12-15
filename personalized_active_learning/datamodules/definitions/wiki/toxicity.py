from personalized_active_learning.datamodules.definitions.wiki.base import WikiDataModule
from typing import List
from pathlib import Path


class ToxicityDataModule(WikiDataModule):
    """The wiki data module for toxicity classification task."""

    @property
    def classes_dimensions(self) -> List[int]:
        return [2]

    @property
    def annotation_columns(self) -> List[str]:
        return ["toxicity"]

    @property
    def embeddings_path(self) -> Path:
        return (
            self.data_dir / f"embeddings/rev_id_to_emb_{self.stratification_type}_toxic.p"
        )