import pandas as pd
import numpy as np
from active_learning.algorithms.base import TextSelectorBase
from active_learning.algorithms.utils import (
    stratify_by_users,
    stratify_by_users_decorator,
)


class RandomSelector(TextSelectorBase):
    def select_annotations(
        self,
        texts: pd.DataFrame,
        amount: int,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: np.ndarray,
    ):
        amount = min(amount, len(not_annotated.index))
        if self.amount_per_user:
            return stratify_by_users(
                not_annotated.sample(frac=1.0), amount_per_user=self.amount_per_user
            )

        return not_annotated.sample(n=amount)


class RandomSelectorDecorated(TextSelectorBase):
    @stratify_by_users_decorator(2)
    def select_annotations(
        self,
        texts: pd.DataFrame,
        amount: int,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: np.ndarray,
    ):
        amount = min(amount, len(not_annotated.index))
        return not_annotated.sample(n=amount)
