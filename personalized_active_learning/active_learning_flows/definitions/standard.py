"""Plain and simple standard active learning flow."""
from personalized_active_learning.active_learning_flows.base import ActiveLearningFlowBase


class StandardActiveLearningFlow(ActiveLearningFlowBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def experiment(
        self,
        max_amount: int,
        step_size: int,
        **kwargs,  # TODO: Leftover to not break compatibility with old code
    ):
        """Run AL.

        Args:
            max_amount: Maximum number of texts that should be annotated before AL is stopped.
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
