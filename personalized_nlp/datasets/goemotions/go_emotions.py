from typing import List
import pandas as pd
import os

from personalized_nlp.settings import STORAGE_DIR, GO_EMOTIONS_LABELS
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule

class GoEmotionsDataModule(BaseDataModule):
    def __init__(
            self, 
            split_sizes: List[float] = [0.55, 0.15, 0.15, 0.15],
            normalize=False,
            classification=False,
            min_annotations_per_text=None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = STORAGE_DIR / 'goemotions'
        self.split_sizes = split_sizes
        self.annotation_column = GO_EMOTIONS_LABELS
        self.text_column = 'text'
        self.word_stats_annotation_column = 'admiration'
        self.embeddings_path = STORAGE_DIR / \
                               f'goemotions/embeddings/text_id_to_emb_{self.embeddings_type}.p'

        self.train_split_names = ['present', 'past']
        self.val_split_names = ['future1']
        self.test_split_names = ['future2']

        self.classification = classification
        self.normalize = normalize
        self.min_annotations_per_text = min_annotations_per_text

        os.makedirs(self.data_dir / 'embeddings', exist_ok=True)

    @property
    def class_dims(self):
        return 2*28

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        self.data = pd.read_csv(
            self.data_dir / 'texts' /'text_data.csv').dropna()
        self.annotations = pd.read_csv(
            self.data_dir / 'texts' /'annotation_data.csv')

        annotated_text_ids = self.annotations.text_id.values
        self.data = self.data.loc[self.data.text_id.isin(annotated_text_ids)].reset_index(False)

        if self.min_annotations_per_text is not None:
            text_id_value_counts = self.annotations.text_id.value_counts()
            text_id_value_counts = text_id_value_counts[text_id_value_counts >= self.min_annotations_per_text]
            self.annotations = self.annotations.loc[self.annotations.text_id.isin(text_id_value_counts.index.tolist())]

        if self.normalize:
            self.normalize_labels()
        self._assign_splits()

        personal_df = self.annotations_with_data.loc[self.annotations_with_data.split == 'past']
        self.compute_annotator_biases(personal_df)

    def normalize_labels(self):
        annotation_column = self.annotation_column
        df = self.annotations

        mins = df.loc[:, annotation_column].values.min(axis=0)
        df.loc[:, annotation_column] = (df.loc[:, annotation_column] - mins)

        maxes = df.loc[:, annotation_column].values.max(axis=0)
        df.loc[:, annotation_column] = df.loc[:, annotation_column] / maxes

    def _assign_splits(self):
        self.data = split_texts(self.data, self.split_sizes)
