import random
import numpy as np
import torch
import os
import pickle
from pytorch_lightning import loggers as pl_loggers
from itertools import product

from personalized_nlp.settings import LOGS_DIR
from personalized_nlp.learning.train import train_test
from personalized_nlp.models.annotator_word import AnnotatorWordEmbeddingNet
from personalized_nlp.models.annotator import AnnotatorEmbeddingNet
from personalized_nlp.models.bias import AnnotatorBiasNet
from personalized_nlp.models.human_bias import HumanBiasNet
from personalized_nlp.models.onehot import NetOneHot
from personalized_nlp.models.baseline import Net
from personalized_nlp.datasets.jester.jester import JesterDataModule

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["WANDB_START_METHOD"] = "thread"


def seed_everything():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


if __name__ == '__main__':
    seed_everything()

    results = []
    regression = True
    min_word_counts = [5]
    words_per_texts = [1]
    dp_embs = [0.25]
    
    #for embeddings_type in ['labse', 'mpnet', 'random']:
    for min_word_count, words_per_text, dp_emb, in product(min_word_counts, words_per_texts, dp_embs):
        #for embeddings_type in ['mpnet']:
        #for embeddings_type in ['xlmr', 'labse', 'mpnet', 'random']:
        for embeddings_type in ['xlmr', 'bert', 'deberta', 'labse', 'mpnet', 'random']:
            data_module = JesterDataModule(embeddings_type=embeddings_type, normalize=regression,
                                            batch_size=3000)
            data_module.prepare_data()
            data_module.setup()
            data_module.compute_word_stats(
                min_word_count=min_word_count,
                min_std=0.0,
                words_per_text=words_per_text
            )

            for model_type in ['baseline', 'peb', 'bias', 'embedding', 'word_embedding']:
                for embedding_dim in [50]:
                    for fold_num in range(10):

                        hparams = {
                            'dataset': type(data_module).__name__,
                            'model_type': model_type,
                            'embeddings_type': embeddings_type,
                            'embedding_size': embedding_dim,
                            'fold_num': fold_num,
                            'regression': regression,
                            'words_per_texts': words_per_text,
                            'min_word_count': min_word_count,
                            'dp_emb': dp_emb
                        }

                        logger = pl_loggers.WandbLogger(
                            save_dir=LOGS_DIR, config=hparams, project='Jester_fixed_limit_past', 
                            log_model=False)

                        output_dim = len(data_module.class_dims)
                        text_embedding_dim = data_module.text_embedding_dim

                        if model_type == 'baseline':
                            model = Net(output_dim=output_dim,
                                        text_embedding_dim=text_embedding_dim)
                        elif model_type == 'onehot':
                            model = NetOneHot(output_dim=output_dim, annotator_num=data_module.annotators_number,
                                            text_embedding_dim=text_embedding_dim)
                        elif model_type == 'peb':
                            model = HumanBiasNet(output_dim=output_dim, bias_vector_length=len(data_module.class_dims),
                                                text_embedding_dim=text_embedding_dim)
                        elif model_type == 'bias':
                            model = AnnotatorBiasNet(output_dim=output_dim, text_embedding_dim=text_embedding_dim,
                                                    word_num=data_module.words_number, annotator_num=data_module.annotators_number)
                        elif model_type == 'embedding':
                            model = AnnotatorEmbeddingNet(output_dim=output_dim, text_embedding_dim=text_embedding_dim, word_num=data_module.words_number,
                                                        annotator_num=data_module.annotators_number, dp=0.0, dp_emb=dp_emb,
                                                        embedding_dim=embedding_dim, hidden_dim=100)
                        elif model_type == 'word_embedding':
                            model = AnnotatorWordEmbeddingNet(output_dim=output_dim, text_embedding_dim=text_embedding_dim, word_num=data_module.words_number,
                                                            annotator_num=data_module.annotators_number, dp=0.0, dp_emb=dp_emb,
                                                            embedding_dim=embedding_dim, hidden_dim=100)

                        train_test(data_module, model, epochs=6, lr=0.008, regression=regression,
                                use_cuda=True, logger=logger)

                        logger.experiment.finish()