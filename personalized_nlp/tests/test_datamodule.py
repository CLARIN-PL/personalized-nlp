import unittest
from itertools import product
from personalized_nlp.datasets.wiki.aggression import AggressionDataModule


class TestDatamoduleInit(unittest.TestCase):
    def test_datamodule_parameters(self):
        major_votings = [False, True]
        normalizes = [True, False]
        past_annotations_limits = [0, 5, 10]
        stratify_folds_bys = ["texts", "users"]
        test_folds = [0, 5, 9]
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

            self.assertGreater(len(data_module.annotations.index), 0)
            self.assertGreater(len(data_module.data.index), 0)
