import os

from pathlib import Path
from typing import List

import pandas as pd

from settings import DATA_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class KrDataModule(BaseDataModule):
    @property
    def annotations_file(self) -> str:
        return f"kr_{self.dataset_num}_annotations_{self.stratify_folds_by}_folds.csv"

    @property
    def data_file(self) -> str:
        return f"kr_{self.dataset_num}_data_processed.csv"

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "stance_detection"

    def __init__(
        self,
        dataset_num: int = 1,
        **kwargs,
    ):
        self.dataset_num = dataset_num
        super().__init__(**kwargs)

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.data_dir / self.data_file)
        self.annotations = pd.read_csv(self.data_dir / self.annotations_file)
        self.annotations["annotator_id"] = self.annotations["user_id"]

    @property
    def class_dims(self) -> List[int]:
        return [2] * len(self.annotation_columns)

    @property
    def annotation_columns(self) -> List[str]:
        if self.dataset_num == 1:
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
            ]
        else:
            return [
                "Czy tekst wzbudza w tobie jakiekolwiek emocje?",
                # "Pozytywne (1-5)",
                # "Negatywne (1-5)",
                # "Radość, szczęście (1-5)",
                # "Zachwyt, podziw, duma (1-5)",
                # "Podnosi na duchu, inspiruje (1-5)",
                # "Spokój, relaks (1-5)",
                # "Zaskoczenie, zdziwienie (1-5)",
                # "Współczucie (1-5)",
                # "Strach, niepokój (1-5)",
                # "Smutek, nieszczęście (1-5)",
                # "Wstręt, obrzydzenie (1-5)",
                # "Złość, wkurzenie, gniew, irytacja (1-5)",
                # "Tekst jest: Ironiczny, sarkastyczny (1-5, n)",
                # "Żenujący (1-5, n)",
                # "Wulgarny (1-5, n)",
                # "Polityczny (1-5, n)",
                # "Interesujący, ciekawy (1-5, n)",
                # "Zgadzam się z tekstem (1-5, n)",
                # "Wierzę w tę informację (1-5, n)",
                "Czy tekst ma charakter obraźliwy lub lekceważący?",
                # "Obraża mnie (1-5)",
                # "Może kogoś atakować / obrażać / lekcewazyć (1-5)",
                "W jaki sposób obraża: odczłowieczenie",
                "Mowa nienawiści",
                "Nawoływanie do przemocy",
                "Nawoływanie do ludobójstwa",
                "Niesprawiedliwe uogólnienie, stereotypy",
                "Lekceważenie",
                "Upokorzenie",
                # "Mnie bawi / śmieszy (1-5)",
                # "Może kogoś bawić (1-5)",
                "Tekst śmieszy ze względu na: Czarny humor",
                "rozluźnianie atmosfery",
                "psucie atmosfery",
                "humor sytuacyjny",
                "ironię",
                "sarkazm",
                "dwuznaczność",
                "seksualność",
                "humor fekalny",
                "wyolbrzymienie",
                "kontrast",
                "ma charakter dowcipu, kawału, żartu",
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
                "może kogoś atakować / obrażać",
                "to kogoś sprawiedliwie atakuje",
                "to kogoś niesprawiedliwie atakuje",
            ]

    @property
    def embeddings_path(self) -> Path:
        return (
            self.data_dir
            / f"embeddings/rev_id_to_emb_kr{self.dataset_num}_{self.embeddings_type}.p"
        )
