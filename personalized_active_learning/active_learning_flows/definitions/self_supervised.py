"""Plain and simple standard active learning flow."""
from personalized_active_learning.active_learning_flows.base import ActiveLearningFlowBase


class SelfSupervisedActiveLearningFlow(ActiveLearningFlowBase):
    def __init__(self, train_with_all_annotations: bool = True, **kwargs) -> None:
        """Initialize object.

        Args:
            train_with_all_annotations: Whether model should be additionally trained
                with all annotations as baseline.
            kwargs: Keywords arguments for `ActiveLearningFlowBase`.
        """
        super().__init__(**kwargs)
        self.train_with_all_annotations = train_with_all_annotations

    def experiment(
        self,
        max_amount: int,
        step_size: int,
    ):
        """Run AL.

        Args:
            max_amount: Maximum number of texts that should be annotated before
                AL is stopped.
            step_size: The number of texts that should be annotated in each AL cycle.

        """
        while self.annotated_amount < max_amount:
            not_annotated = (self.dataset.annotations.split == "none").sum()
            if not_annotated == 0:
                break
            self.add_annotations(step_size)
            self.train_model()

        if self.train_with_all_annotations:
            # Train at all annotations as baseline
            not_annotated = (self.dataset.annotations.split == "none").sum()
            if not_annotated > 0:
                self.add_annotations(not_annotated)
                self.train_model()
