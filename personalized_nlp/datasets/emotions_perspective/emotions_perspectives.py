from lib2to3.pytree import Base
from typing import List

import pandas as pd
import os

from personalized_nlp.settings import STORAGE_DIR
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule

class EmotionsPerspectiveDataModule(BaseDataModule):
    def __init__(
            self, 
            language: str = 'english',
            split_sizes: List[float] = [0.55, 0.15, 0.15, 0.15],
            normalize=False,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = STORAGE_DIR / 'emotion_nlp_perspectives'
        self.split_sizes = split_sizes
        self.language = language
        self.annotation_column = ['joy',
                                  'trust',
                                  'anticipation',
                                  'surprise',
                                  'fear',
                                  'sadness',
                                  'disgust',
                                  'anger',
                                  'valence',
                                  'arousal']
        self.text_column = 'text'
        # self.word_stats_annotation_column = 'POBUDZENIE EMOCJONALNE'
        self.embeddings_path = STORAGE_DIR / \
            f'emotion_nlp_perspectives/embeddings/text_id_to_emb_{self.embeddings_type}_{language}.p'

        self.train_split_names = ['present', 'past']
        self.val_split_names = ['future1']
        self.test_split_names = ['future2']

        self.normalize = normalize

        os.makedirs(self.data_dir / 'embeddings', exist_ok=True)

    @property
    def class_dims(self):
        return [4] * 8 + [7, 4]
    
    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        self.data = pd.read_csv(
            self.data_dir / 'texts' /'text_data.csv')
        self.annotations = pd.read_csv(
            self.data_dir / 'texts' /'annotation_data.csv')

        if self.normalize:
            self.normalize_labels()
        self.assign_splits()

        personal_df = self.annotations_with_data.loc[self.annotations_with_data.split == 'past']
        self.compute_annotator_biases(personal_df)

    def normalize_labels(self):
        annotation_column = self.annotation_column
        df = self.annotations