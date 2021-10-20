from typing import Optional, List

import pandas as pd
from torch.utils.data import DataLoader

from personalized_nlp.settings import STORAGE_DIR
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class UnhealthyDataModule(BaseDataModule):
    def __init__(
            self,
            data_dir: str = STORAGE_DIR / 'unhealthy_conversations/',
            split_sizes: List[float] = [0.55, 0.15, 0.15, 0.15],
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.folds_num = 10
        self.data_dir = data_dir
        self.data_path = self.data_dir / 'unhealthy_full.csv'
        self.split_sizes = split_sizes
        self.annotation_column = ['antagonize',
                                  'condescending',
                                  'dismissive',
                                  'generalisation',
                                  'generalisation_unfair',
                                  'healthy',
                                  'hostile',
                                  'sarcastic']
        self.text_column = 'text'

        self.word_stats_annotation_column = 'healthy'
        self.embeddings_path = STORAGE_DIR / \
            f'unhealthy_conversations/embeddings/text_id_to_emb_{self.embeddings_type}.p'

        self.train_split_names = ['present', 'past']
        self.val_split_names = ['future1']
        self.test_split_names = ['future2']

    @property
    def class_dims(self):
        return [2] * 8

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        full_data = pd.read_csv(self.data_dir / 'unhealthy_full.csv').dropna()
        self.data = full_data.loc[:, ['_unit_id', 'comment']].drop_duplicates().reset_index(drop=True)
        self.data.columns = ['text_id', 'text']
        
        self.annotations = full_data.loc[:, ['_unit_id', '_worker_id'] + self.annotation_column]
        self.annotations.columns = ['text_id', 'annotator_id'] + self.annotation_column

        self._assign_splits()
        
        personal_df = self.annotations_with_data.loc[self.annotations_with_data.split == 'past']
        self.compute_annotator_biases(personal_df)

    def _assign_splits(self):
        sizes = [0.55, 0.15, 0.15, 0.15]
        self.data = split_texts(self.data, sizes)
