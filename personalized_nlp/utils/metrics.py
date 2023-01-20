from typing import Any

import torchmetrics
from torch import Tensor


class F1Class(torchmetrics.classification.MulticlassF1Score):
    def __init__(self, class_idx: int = -1, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.class_idx = class_idx

    def compute(self) -> Tensor:
        f1_for_classes = super().compute()

        return f1_for_classes[self.class_idx]


class PrecisionClass(torchmetrics.classification.MulticlassPrecision):
    def __init__(self, class_idx: int = -1, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.class_idx = class_idx

    def compute(self) -> Tensor:
        precision_for_classes = super().compute()

        return precision_for_classes[self.class_idx]


class RecallClass(torchmetrics.classification.MulticlassRecall):
    def __init__(self, class_idx: int = -1, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.class_idx = class_idx

    def compute(self) -> Tensor:
        recall_for_classes = super().compute()

        return recall_for_classes[self.class_idx]
