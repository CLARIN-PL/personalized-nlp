import os

from pathlib import Path
from typing import List, Optional

import pandas as pd

from settings import DATA_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class DoccanoDataModule(BaseDataModule):
    @property
    def annotations_file(self) -> str:
        return f"annotations_{self.stratify_folds_by}_folds.csv"

    @property
    def data_file(self) -> str:
        return f"data_processed.csv"

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "doccano"

    def __init__(
        self,
        empty_annotations_strategy: Optional[str] = None,
        annotations_number: Optional[int] = None,
        texts_number: Optional[int] = None,
        min_annotations_per_text: Optional[int] = None,
        randomize_questionnaries: bool = False,
        used_questionnare_idx: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            empty_annotations_strategy (str, optional): What to do with empty (-1) annotations.
            If None, the text with any number of empty task annotation is still used in learning process,
            but it is ignored in loss/metric calculation. If "drop", the annotations with
            any empty task annotations in traning dataset are dropped. Defaults to None.
        """
        self.empty_annotations_strategy = empty_annotations_strategy
        self.annotations_number = annotations_number
        self.texts_number = texts_number
        self.min_annotations_per_text = min_annotations_per_text
        self.randomize_questionnaries = randomize_questionnaries

        self.used_questionnare_idx = used_questionnare_idx
        self.used_questionnare_name: Optional[str] = None

        super().__init__(**kwargs)

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def load_questionnaries(self):
        annotations = self.annotations
        user_id_df = annotations[["user_id", "user_id_original"]].drop_duplicates()
        user_id_df = user_id_df.set_index("user_id_original")
        user_id_mapping_dict = user_id_df.to_dict()["user_id"]

        q_answers = pd.read_csv(self.data_dir / "doccano_questionnaires.csv")
        df = q_answers
        df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
        df.iloc[:, 1:] = (df.iloc[:, 1:] - df.iloc[:, 1:].mean(axis=0)) / df.iloc[
            :, 1:
        ].std(axis=0)

        df["user_id"] = df["user_id"].map(user_id_mapping_dict)

        self.q_answers = df.rename(columns={"user_id": "annotator_id"})
        self.q_answers = self.q_answers.set_index("annotator_id").sort_index()

        if self.used_questionnare_idx is not None:
            self.used_questionnare_name = self.q_answers.columns.tolist()[
                self.used_questionnare_idx
            ]
            self.q_answers = self.q_answers.iloc[:, [self.used_questionnare_idx]]
            print(self.q_answers)

        if self.randomize_questionnaries:
            self.q_answers = self.q_answers.sample(frac=1.0)
            self.q_answers.index = range(len(self.q_answers))

    def _get_annotator_features(self):
        features = super()._get_annotator_features()

        features["q_answers"] = self.q_answers.values.astype(float)

        return features

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.data_dir / self.data_file)
        self.annotations = pd.read_csv(self.data_dir / self.annotations_file)
        self.annotations["annotator_id"] = self.annotations["user_id"]

        if self.empty_annotations_strategy == "drop":
            any_empty_annotation_mask = (
                self.annotations[self.annotation_columns] == -1
            ).any(axis=1)
            train_fold_mask = self.annotations.fold.isin(self.train_folds)
            self.annotations = self.annotations.loc[
                ~(any_empty_annotation_mask & train_fold_mask)
            ].reset_index(drop=True)

        if self.min_annotations_per_text is not None:
            text_id_value_counts = self.annotations.text_id.value_counts()
            text_id_value_counts = text_id_value_counts[
                text_id_value_counts >= self.min_annotations_per_text
            ]
            good_text_ids = text_id_value_counts.index.tolist()
            self.annotations = self.annotations.loc[
                self.annotations.text_id.isin(good_text_ids)
            ]

        if not self.regression:

            def get_class_label(val):
                if val == 0:
                    return 0
                elif val > 0:
                    return 1
                else:
                    return -1

            for col in self.annotation_columns:
                self.annotations[col] = self.annotations[col].apply(get_class_label)
        else:
            for col in self.annotation_columns:
                self.annotations[col] = self.annotations[col].clip(0, 10)

        self.load_questionnaries()

    def _after_setup(self):
        if not self.annotations_number:
            return

        df = self.annotations.copy()
        df["original_index"] = df.reset_index()["index"].values
        df["annotation_idx_"] = (
            df.groupby("text_id")
            .apply(lambda rows: rows.reset_index().reset_index().sample(frac=1.0))[
                "level_0"
            ]
            .values
        )

        text_ids = df["text_id"].drop_duplicates().sort_values()[: self.texts_number]

        df = df.loc[df.text_id.isin(text_ids)].sort_values(by="annotation_idx_")
        df = df[: self.annotations_number]
        self.annotations.loc[self.annotations.split == "train", "split"] = "None"
        self.annotations.loc[df["original_index"].tolist(), "split"] = "train"

    @property
    def class_dims(self) -> List[int]:
        return [2] * len(self.annotation_columns)

    @property
    def annotation_columns(self) -> List[str]:
        return [
            "Pozytywne",
            "Negatywne",
            "Radość",
            "Zachwyt",
            "Inspiruje",
            "Spokój",
            "Zaskoczenie",
            "Współczucie",
            "Strach",
            "Smutek",
            "Wstręt",
            "Złość",
            "Ironiczny",
            "Żenujący",
            "Wulgarny",
            "Polityczny",
            "Interesujący",
            "Zrozumiały",
            "Potrzebuję więcej informacji, aby ocenić ten tekst",
            "Obraża mnie",
            "Może kogoś atakować / obrażać / lekceważyć",
            "Mnie bawi/śmieszy?",
            "Może kogoś bawić?",
            "Zgadzam się z tekstem",
            "Wierzę w tę informację",
            "Czuję sympatię do autora",
        ]

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / f"embeddings/rev_id_to_emb_sd_{self.embeddings_type}.p"
