from typing import Optional

from personalized_active_learning.algorithms.self_supervised_learning import ISelfSupervisedTrainer
from personalized_active_learning.datamodules import BaseDataModule
from personalized_active_learning.models import IModel


class SampledBatchTrainer(ISelfSupervisedTrainer):
    """Perform self supervised learning with sampling from
    both real labels and pseudo-labels.

    As defined in papers:
    "Rethinking deep active learning: Using unlabeled data at model training".
    "Label Propagation for Deep Semi-supervised Learning"

    """

    def __init__(
        self,
        sampling_ratio: float,  # TODO ratio or weights?
        batch_size: int,
        wandb_project_name: str,
        random_seed: Optional[int] = None,
        num_workers: int = 4,
        lr: float = 1e-2,
        number_of_epochs: int = 6,
        use_cuda: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self._wandb_project_name = wandb_project_name
        self._sampling_ratio = sampling_ratio
        self._batch_size = batch_size
        self._random_seed = random_seed
        self._num_workers = num_workers
        self._lr = lr
        self._number_of_epochs = number_of_epochs
        self._use_cuda = use_cuda
        self._random_seed = seed

    def train(
        self,
        dataset: BaseDataModule,
        model: IModel,
        pseudo_labels,
        unlabelled_indexes,
    ) -> IModel:
        raise NotImplementedError
