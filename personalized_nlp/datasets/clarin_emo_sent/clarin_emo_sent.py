import os

from pathlib import Path
from typing import List

import pandas as pd

from settings import DATA_DIR
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class ClarinEmoSentDataModule(BaseDataModule):
    
    """Datamodule for Clarin Emo Sentiment Dataset. 
    
    This is the version with multiple user-text annotations

    """
    @property 
    def annotations_file(self) -> str:
        return f'annotations_{self.stratify_folds_by}_folds.csv'
    
    @property 
    def data_file(self) -> str:
        return f'data_processed.csv'

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "clarin_emo_sent"

    def __init__(
        self,
        **kwargs,
    ):
        super(ClarinEmoSentDataModule, self).__init__(**kwargs)

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def _remap_column_names(self, df):
        mapping = {"user_id": "annotator_id"}
        df.columns = [mapping.get(col, col) for col in df.columns]
        return df

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.data_dir / self.data_file)
        self.data = self._remap_column_names(self.data)

        self.annotations = pd.read_csv(self.data_dir / self.annotations_file)
        self.annotations = self._remap_column_names(self.annotations)

    @property
    def class_dims(self) -> List[int]:
        return [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    @property
    def annotation_columns(self) -> List[str]:
        return [
            'radość', 'zaufanie', 'przeczuwanie', 'zdziwienie', 'strach',
            'smutek', 'wstręt', 'gniew', 'pozytywny', 'negatywny', 'neutralny'
        ]

    @property
    def embeddings_path(self) -> Path:
        return (self.data_dir /
                f"embeddings/rev_id_to_emb_{self.embeddings_type}.p")
        
        
        
class ClarinEmoSentNoNoiseDataModule(ClarinEmoSentDataModule):
    """Version with no noise!

    """

    def __init__(
        self,
        **kwargs,
    ):
        super(ClarinEmoSentNoNoiseDataModule, self).__init__(**kwargs)
        
        
    def _strip_redundant(self) -> None:
        """Removes redundant annotations (ex. two annoations of text 202 by user 1) by averaging them.
        """
        annotations = self.annotations
        dim_columns = annotations.columns
        annotations['uid'] = annotations['text_id'].astype(str) + ' ' + annotations['annotator_id'].astype(str)
        grouped_df = annotations.groupby('uid').mean()[dim_columns].round().astype(int).reset_index()
        self.annotations = grouped_df
        

    def prepare_data(self) -> None:
        super().prepare_data()
        self._strip_redundant()
