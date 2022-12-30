from typing import Tuple
from personalized_active_learning.embeddings.personalised.base import (
    PersonalisedEmbeddings
)
import pandas as pd
import swifter  # noqa: F401


class UserIdEmbeddings(PersonalisedEmbeddings):
    """Main idea: Make pairs annotator-text. Append annotator id to text data."""

    @property
    def name(self) -> str:
        return "user_id"

    def apply_personalisation(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        def change_dataset(row) -> pd.Series:
            result = self.data[self.data["text_id"] == row["text_id"]]
            result["text"] += f" {row['annotator_id']}"
            return result

        new_data = self.annotations[
            ["text_id", "annotator_id"]
        ].swifter.apply(change_dataset, axis=1)

        new_df = pd.concat(list(new_data), ignore_index=True)

        new_df["old_text_id"] = new_df["text_id"]
        new_df["text_id"] = new_df.index

        self.annotations["old_text_id"] = self.annotations["text_id"]
        self.annotations["text_id"] = self.annotations.index

        return new_df, self.annotations
