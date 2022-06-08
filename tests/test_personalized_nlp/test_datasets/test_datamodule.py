from itertools import product

import pytest

from personalized_nlp.datasets.wiki import AggressionDataModule


class TestDatamodule:

    # to powinien być test parametryczny, ale nie mam czasu
    def test_datamodule_parameters(self):
        major_votings = [False] #[False, True]
        normalizes = [True] # [True, False]
        past_annotations_limits = [0] #[0, 5, 10]
        stratify_folds_bys = ['texts'] #["texts", "users"]
        test_folds = [9] #[0, 5, 9]
        embeddings_fold = None
        folds_num = 10

        for (
            major_voting,
            normalize,
            past_annotations_limit,
            stratify_folds_by,
            test_fold,
        ) in product(
            major_votings,
            normalizes,
            past_annotations_limits,
            stratify_folds_bys,
            test_folds,
        ):
            data_module = AggressionDataModule(
                embeddings_type="labse",
                major_voting=major_voting,
                folds_num=folds_num,
                normalize=normalize,
                past_annotations_limit=past_annotations_limit,
                stratify_folds_by=stratify_folds_by,
                embeddings_fold=embeddings_fold,
                test_fold=test_fold,
            )

            assert len(data_module.annotations.index) >  0
            assert len(data_module.data.index) > 0
