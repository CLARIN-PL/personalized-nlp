import abc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Optional

from personalized_active_learning.algorithms.label_propagation.interface import ILabelPropagator
from personalized_active_learning.datamodules import BaseDataModule
from personalized_active_learning.models import IModel


class RandomForestPropagator(ILabelPropagator):
    """Perform label propagation using random forest."""

    def __init__(
        self,
        wandb_project_name: str,
        random_seed: int,
        n_estimators: int,
        max_depth: Optional[int] = None,
        use_text_ids_during_propagation: bool = True,
        use_annotator_ids_during_propagation: bool = True,
        num_workers: int = 4,
        use_cuda: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self._wandb_project_name = wandb_project_name
        self._random_seed = random_seed
        self.use_text_ids_during_propagation = use_text_ids_during_propagation
        self.use_annotator_ids_during_propagation = use_annotator_ids_during_propagation
        self._num_workers = num_workers
        self._use_cuda = use_cuda
        self._random_seed = seed
        self._n_estimators = n_estimators
        self._max_depth = max_depth

    @abc.abstractmethod
    def propagate_labels(
        self,
        dataset: BaseDataModule,
        model: Optional[IModel] = None
    ) -> np.ndarray:
        """Use label propagation to get new pseudo labels.

        Args:
            dataset: Dataset containing data used for pretraining.
            model: Model that will be used for label propagation.
                (Can be none as we sometimes create such model inside).
        Returns:
            Pseudo-labels.

        """

        labelled_data,  unlabelled_data = self._prepare_features_for_propagation(dataset)
        labelled_embs = labelled_data['emb'].tolist()
        unlabelled_embs = unlabelled_data['emb'].tolist()

        pseudo_labels_list = []
        for col_name in dataset.annotation_columns:
            col_labels = labelled_data[col_name].tolist()

            rfc = RandomForestClassifier(n_estimators=self._n_estimators, max_depth=self._max_depth,
                                         random_state=self._random_seed)
            rfc.fit(labelled_embs, col_labels)
            col_pseudo_labels = rfc.predict(unlabelled_embs)
            pseudo_labels_list.append(col_pseudo_labels)
        pseudo_labels = np.stack(pseudo_labels_list, axis=-1)
        return labelled_data,  unlabelled_data, pseudo_labels

    def _train_forests(self, annotation_columns, labelled_data):
        random_forest_classifiers = []

        return random_forest_classifiers

    def _prepare_features_for_propagation(
        self,
        dataset: BaseDataModule
    ) -> (pd.DataFrame, pd.DataFrame):
        """Prepares annotated examples with embeddings and
        not annotated examples with embeddings.

        Args:
            dataset: Dataset containing data used for pretraining.
        Returns:
            annotations_emb_train: Embeddings of training examples
            with annotations form specified annotator.
            annotations_emb_none: Embeddings of unlabelled examples
            with annotations form specified annotator.

        """
        annotations = dataset.annotations
        annotations_train = annotations[annotations['split'] == 'train'].copy()
        annotations_none = annotations[annotations['split'] == 'none'].copy()

        def pick_embedding(row):
            row.emb = dataset.text_embeddings[row.text_id]
            return row

        annotations_train['emb'] = None
        annotations_emb_train = annotations_train.apply(pick_embedding, axis=1)

        annotations_none['emb'] = None
        annotations_emb_none = annotations_none.apply(pick_embedding, axis=1)

        return annotations_emb_train, annotations_emb_none
