import os
from itertools import product

from personalized_nlp.learning.train import train_test
from personalized_nlp.models import models as models_dict
from personalized_nlp.settings import LOGS_DIR
from personalized_nlp.datasets.humor.humor import HumorDataModule
from personalized_nlp.utils import seed_everything
from pytorch_lightning import loggers as pl_loggers

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    regression = True
    datamodule_cls = HumorDataModule
    embedding_types = ["labse", "mpnet", "xlmr", "random"]
    model_types = ["baseline", "onehot", "peb", "bias", "embedding", "word_embedding"]
    wandb_project_name = "Humor_exp"
    limit_past_annotations_list = [None]  # range(20)
    fold_nums = 2
    min_annotations_per_text = 2

    min_word_counts = [50]
    words_per_texts = [15]

    batch_size = 3000
    dp_embs = [0.25]
    embedding_dims = [50]
    epochs = 50
    lr_rate = 0.008

    use_cuda = True

    for (
        min_word_count,
        words_per_text,
        embeddings_type,
        limit_past_annotations,
    ) in product(
        min_word_counts, words_per_texts, embedding_types, limit_past_annotations_list
    ):

        seed_everything()
        data_module = datamodule_cls(
            embeddings_type=embeddings_type,
            normalize=regression,
            batch_size=batch_size,
            min_annotations_per_text=min_annotations_per_text,
            past_annotations_limit=limit_past_annotations,
        )
        data_module.prepare_data()
        data_module.setup()
        data_module.compute_word_stats(
            min_word_count=min_word_count,
            min_std=0.0,
            words_per_text=words_per_text,
        )

        for model_type, embedding_dim, dp_emb, fold_num in product(
            model_types, embedding_dims, dp_embs, range(fold_nums)
        ):
            hparams = {
                "dataset": type(data_module).__name__,
                "model_type": model_type,
                "embeddings_type": embeddings_type,
                "embedding_size": embedding_dim,
                "fold_num": fold_num,
                "regression": regression,
                "words_per_texts": words_per_text,
                "min_word_count": min_word_count,
                "dp_emb": dp_emb,
            }

            logger = pl_loggers.WandbLogger(
                save_dir=LOGS_DIR,
                config=hparams,
                project=wandb_project_name,
                log_model=False,
            )

            output_dim = (
                len(data_module.class_dims)
                if regression
                else sum(data_module.class_dims)
            )
            text_embedding_dim = data_module.text_embedding_dim
            model_cls = models_dict[model_type]

            model = model_cls(
                output_dim=output_dim,
                text_embedding_dim=text_embedding_dim,
                word_num=data_module.words_number,
                annotator_num=data_module.annotators_number,
                dp=0.0,
                dp_emb=dp_emb,
                embedding_dim=embedding_dim,
                hidden_dim=100,
                bias_vector_length=len(data_module.class_dims),
            )

            train_test(
                data_module,
                model,
                epochs=epochs,
                lr=lr_rate,
                regression=regression,
                use_cuda=use_cuda,
                logger=logger,
                test_fold=fold_num,
            )

            logger.experiment.finish()
