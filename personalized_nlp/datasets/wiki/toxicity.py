from personalized_nlp.datasets.wiki.base import WikiDataModule
from personalized_nlp.settings import STORAGE_DIR, TOXICITY_URL


class ToxicityDataModule(WikiDataModule):
    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_url = TOXICITY_URL

        self.annotation_column = 'toxicity'
        self.embeddings_path = STORAGE_DIR / \
            f'wiki_data/embeddings/rev_id_to_emb_{self.embeddings_type}_toxic.p'
