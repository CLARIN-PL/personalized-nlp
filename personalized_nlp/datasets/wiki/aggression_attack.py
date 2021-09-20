import pandas as pd
import pickle
import torch

from personalized_nlp.datasets.wiki.base import WikiDataModule
from personalized_nlp.settings import STORAGE_DIR, AGGRESSION_URL
from personalized_nlp.utils.biases import get_annotator_biases

class AggressionAttackDataModule(WikiDataModule):
    def __init__(
            self,
            data_dir: str = STORAGE_DIR / 'wiki_data',
            batch_size: int = 3000,
            embeddings_path: str = STORAGE_DIR / 'embeddings/rev_id_to_emb_bert_aggression.p',
            **kwargs,
    ):
        super().__init__(data_dir, batch_size, **kwargs)

        self.data_path = self.data_dir / 'aggression_annotations.tsv'
        self.data_url = AGGRESSION_URL

        self.annotation_column = ['aggression', 'attack']
        self.word_stats_annotation_column = 'aggression'
        self.embeddings_path = embeddings_path

    @property
    def class_dims(self):
        return [2, 2]

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.data_dir / 'aggression_annotated_comments.tsv', sep='\t')
        self.annotations = pd.read_csv(self.data_dir / 'aggression_annotations.tsv', sep='\t')
        self.annotators = pd.read_csv(self.data_dir / 'aggression_worker_demographics.tsv', sep='\t')

        attack_annotations = pd.read_csv(self.data_dir / 'attack_annotations.tsv', sep='\t')
        self.annotations = self.annotations.merge(attack_annotations)
        
        text_idx_to_emb = pickle.load(open(self.embeddings_path, 'rb'))
        embeddings = []
        for text_idx in range(len(text_idx_to_emb.keys())):
            embeddings.append(text_idx_to_emb[text_idx])

        self.text_embeddings = torch.tensor(embeddings)
    
    def compute_annotator_biases(self, personal_df: pd.DataFrame):
        annotator_id_df = pd.DataFrame(self.annotations.annotator_id.unique(), columns=['annotator_id'])

        annotator_biases = get_annotator_biases(personal_df, self.annotation_column)
        annotator_biases = annotator_id_df.merge(annotator_biases.reset_index(), how='left')
        self.annotator_biases = annotator_biases.set_index('annotator_id').sort_index().fillna(0)