from personalized_nlp.datasets.wiki.base import WikiDataModule
from personalized_nlp.settings import PROJECT_DIR, STORAGE_DIR, ATTACK_URL

class AttackDataModule(WikiDataModule):
    def __init__(
            self,
            data_dir: str = STORAGE_DIR / 'wiki_data',
            batch_size: int = 3000,
            embeddings_type: str = 'bert',
            **kwargs,
    ):
        super().__init__(data_dir, batch_size, embeddings_type, **kwargs)

        self.data_path = self.data_dir / 'attack_annotations.tsv'
        self.data_url = ATTACK_URL

        self.annotation_column = 'attack'
        self.embeddings_path = STORAGE_DIR / f'wiki_data/embeddings/rev_id_to_emb_{embeddings_type}_attack.p'