from personalized_nlp.datasets.wiki.base import WikiDataModule
from typing import List
from pathlib import Path


class AgressionAttackCombinedDatamodule(WikiDataModule):
    
    @property
    def worker_demographics_file(self) -> str:
        return f'aggression_worker_demographics.tsv'
    
    @property 
    def annotations_file(self) -> str:
        return f'agression_attack_{self.stratify_folds_by}_folds.csv'
    
    @property 
    def data_file(self) -> str:
        return f'attack_annotated_comments_processed.csv'
    
    @property
    def class_dims(self) -> List[int]:
        return [2, 2]

    @property
    def annotation_columns(self) -> List[str]:
        return ["attack", "aggression"]

    @property
    def embeddings_path(self) -> Path:
        return (
            self.data_dir / f"embeddings/rev_id_to_emb_{self.embeddings_type}_attack_agression.p"
        )