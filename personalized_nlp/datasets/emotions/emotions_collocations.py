from typing import List

import pandas as pd
import os
from pathlib import Path

from settings import DATA_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class EmotionsCollocationsDatamodule(BaseDataModule):
    
    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "emotions_colocations_data"
    
    @property
    def annotation_columns(self):
        return [
           "VAL",
           "ARO",
           "ANG",
           "DIS",
           "FEA",
           "SAD",
           "ANT",
           "HAP",
           "SUR",
           "TRU"
        ]
        
    @property
    def embeddings_path(self):
        return self.data_dir / "embeddings"
    
    @property 
    def annotations_file(self) -> str:
        return f'cawi1_6000_annotations_normalized_{self.stratify_folds_by}_folds.csv'
    
    @property 
    def data_file(self) -> str:
        return f'texts_processed.csv'

    def __init__(
        self, **kwargs,
    ):
        super().__init__(**kwargs)
        self.language = 'polish'
        
        
    @property
    def class_dims(self):
        return [5] * 8 + [7, 5]


    def prepare_data(self) -> None:
        reanme_map = {'plWordNet ID': 'text_id', 'phrase': 'text', 'participant': 'annotator_id'}
        self.data = pd.read_csv(self.data_dir / self.data_file)
        self.data.rename(columns=reanme_map, inplace=True)

        self.annotations = pd.read_csv(self.data_dir / self.annotations_file).dropna()
        self.annotations.rename(columns=reanme_map, inplace=True)
  