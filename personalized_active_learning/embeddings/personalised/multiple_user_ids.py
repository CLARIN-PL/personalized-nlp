from typing import Tuple
from personalized_active_learning.embeddings.personalised.base import (
    PersonalisedEmbeddings
)
import pandas as pd
import swifter  # noqa: F401


class MultipleUserIdsEmbeddings(PersonalisedEmbeddings):
    """Main idea: Add all annotators ids' to the each row of text data"""

    @property
    def name(self) -> str:
        return "multiple_user_ids"

    def apply_personalisation(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.data["text"] = self.data.swifter.apply(
            lambda row: row["text"] + ' ' + " ".join(
                map(str, list(
                    self.annotations[
                        self.annotations["text_id"] == row["text_id"]
                    ]["annotator_id"]
                ))
            ),
            axis=1
        )
        return self.data, self.annotations
