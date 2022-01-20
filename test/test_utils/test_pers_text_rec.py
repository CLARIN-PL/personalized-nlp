import unittest
from parameterized import parameterized

import pandas as pd

from personalized_nlp.utils.pers_text_rec import get_annotations, _get_first_annotations, _get_next_annotations



class TestGetAnnotations(unittest.TestCase):

    def setUp(self) -> None:
        self.data = pd.DataFrame({
            'text_id': [0, 1, 2, 3, 4, 5],
            'text': ['a', 'aa', 'aaa', 'aaaa', 'aaaaa', 'aaaaaa'],
            'split': ['train', 'train', 'train', 'val', 'val', 'test']
        })
        self.annotations = pd.DataFrame({
            'annotator_id': [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6,
            'text_id': list(self.data['text_id']) * 4
        })
        self.annotators_data = pd.DataFrame({
            'annotator_id': [i for i in range(4)],
        })
        def _first_annotation_rule(user_id: int, annotations: pd.DataFrame) -> pd.DataFrame:
            for i, row in annotations.iterrows():
                if row['annotator_id'] == user_id:
                    return i
        def _next_annotations_rule(user_id: int, annotations: pd.DataFrame) -> pd.DataFrame:
            for i, row in annotations.iterrows():
                if row['user_annotation_order'] == -1 and row['annotator_id'] == user_id: 
                    return i
        self.first_annotation_rule = _first_annotation_rule
        self.next_annotations_rule = _next_annotations_rule

    # test, czy przechodzi zwykły przypadek
    @parameterized.expand([
        [1],
        [2],
        [3]
    ])
    def test_get_annotations_standard_case(self, max_annotations_per_user: int):
        result = get_annotations(
                data = self.data,
                annotations = self.annotations,
                first_annotation_rule=self.first_annotation_rule,
                next_annotations_rule=self.next_annotations_rule,
                max_annotations_per_user=max_annotations_per_user,
            )  
        # merged_original = self.data.merge(self.annotations, on='text_id')
        merged_result = self.data.merge(result, on='text_id')
        self.assertListEqual(
            list(merged_result[merged_result['split'] == 'train']['text_id']),
            list(merged_result[merged_result['split'] == 'train']['user_annotation_order'])
        )
        self.assertEqual(
            len(result[result['user_annotation_order'] != -1]),
            max_annotations_per_user * len(self.annotators_data)
        )
        self.assertSetEqual(
            set(merged_result[merged_result['split'] != 'train']['user_annotation_order']),
            {-1}
        )

    # test, czy wywala błąd na n < 1
    @parameterized.expand([
        [-5],
        [-1],
        [0]
    ])
    def test_get_annotations_max_annotations_per_user_lesser_than_one(self, max_annotations_per_user: int):
        with self.assertRaises(AssertionError):
            _ = get_annotations(
                data = self.data,
                annotations = self.annotations,
                first_annotation_rule=self.first_annotation_rule,
                next_annotations_rule=self.next_annotations_rule,
                max_annotations_per_user=max_annotations_per_user,
            )   



if __name__ == '__main__':
    unittest.main()
