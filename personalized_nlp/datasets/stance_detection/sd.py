from typing import List

import pandas as pd
import os
from pathlib import Path

from settings import DATA_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class SDStanceDetectionDataModule(BaseDataModule):
    @property
    def annotations_file(self) -> str:
        return f"sd_annotations_{self.stratify_folds_by}_folds.csv"

    @property
    def data_file(self) -> str:
        return "sd_data_processed.csv"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / f"embeddings/text_id_to_emb_{self.embeddings_type}.p"

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "stance_detection"

    @property
    def annotation_columns(self) -> List[str]:
        return [
            "cieszy mnie",
            "budzi we mnie zaufanie",
            "spodziewanie się czegoś",
            "zaskoczyło mnie",
            "boję się",
            "smuci mnie",
            "budzi we mnie wstręt",
            "złości mnie",
            "czuję się pozytywnie",
            "czuję się negatywnie",
            "czuję się neutralnie",
            "zgadzam się z tekstem",
            "nie zgadzam się z tekstem",
            "wierzę w tę informację",
            "nie wierzę w tę informację",
            "współczuję",
            "podnosi mnie na duchu",
            "daje mi nadzieję",
            "to jest ironiczne",
            "w tekście jest sarkazm",
            "bawi mnie",
            "ten żart mnie nie bawi",
            "w tekście jest czarny humor",
            "to żenujące",
            "obraża mnie",
            "może kogoś obrażać",
            "to kogoś sprawiedliwie atakuje",
            "to kogoś niesprawiedliwie atakuje",
            # "pro",
            # "anty",
            # "może kogoś atakować / obrażać",
        ]

    @property
    def class_dims(self):
        return [2] * len(self.annotation_columns)

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.data_dir / self.data_file).dropna()
        self.annotations = pd.read_csv(self.data_dir / self.annotations_file)
        self.annotations["annotator_id"] = self.annotations["user_id"]

        # self.annotations = self.annotations.dropna(axis=0)

        # for col in self.annotation_columns:
        #     self.annotations[col] = self.annotations[col].apply(
        #         lambda x: 1 if x == True or x == 1.0 else 0
        #     )
