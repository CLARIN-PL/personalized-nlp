from typing import Tuple
from personalized_active_learning.embeddings.personalised.base import (
    PersonalisedEmbeddings
)
import pandas as pd
import swifter  # noqa: F401


class TokenizedMultipleUserIdsEmbeddings(PersonalisedEmbeddings):
    """Main idea: Add all tokenized annotators ids' to the each row of text data"""

    def __init__(self, texts_data: pd.DataFrame, annotations: pd.DataFrame) -> None:
        super().__init__(texts_data, annotations)
        self._special_tokens = []

    @property
    def name(self) -> str:
        return "tokenized_multiple_user_ids"

    def apply_personalisation(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        def change_dataset(row) -> pd.Series:
            result = row["text"]
            annotators_ids = self.annotations[
                self.annotations["text_id"] == row["text_id"]
            ]["annotator_id"]
            for annotator_id in list(annotators_ids):
                user_id_tokenized = f"_#{str(annotator_id)}#_"
                self._special_tokens.append(user_id_tokenized)
                result += f" {user_id_tokenized}"
            return result

        self.data["text"] = self.data.swifter.apply(
            change_dataset,
            axis=1
        )
        return self.data, self.annotations
