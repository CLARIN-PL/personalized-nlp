from personalized_nlp.datasets.wiki.base import WikiDataModule
from personalized_nlp.settings import PROJECT_DIR, STORAGE_DIR, AGGRESSION_URL

from typing import List


class AggressionDataModule(WikiDataModule):
    def __init__(
            self,
            data_dir: str = STORAGE_DIR / 'wiki_data',
            **kwargs,
    ):
        super().__init__(data_dir, **kwargs)

        self.data_path = self.data_dir / 'aggression_annotations.tsv'
        self.data_url = AGGRESSION_URL

        self.annotation_column = 'aggression'
        self.embeddings_path = STORAGE_DIR / \
            f'wiki_data/embeddings/rev_id_to_emb_{self.embeddings_type}_aggression.p'
