import os
from itertools import product

from personalized_nlp.learning.train import train_test
from personalized_nlp.models import models as models_dict
from personalized_nlp.settings import LOGS_DIR
from personalized_nlp.datasets.wiki.toxicity import ToxicityDataModule
from personalized_nlp.datasets.wiki.attack import AttackDataModule
from personalized_nlp.datasets.wiki.aggression import AggressionDataModule
from personalized_nlp.utils import seed_everything
import personalized_nlp.utils.callbacks as callbacks
from pytorch_lightning import loggers as pl_loggers
from personalized_nlp.settings import TRANSFORMER_MODEL_STRINGS

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = "Transformer_test"

    regression = False
    datamodule_clses = [AggressionDataModule, AttackDataModule, ToxicityDataModule]
    stratify_by_options = [
        # None,
        # "users",
        "texts",
    ]
    embedding_types = ["labse"]
    # embedding_types = ["xlmr", "bert", "deberta", "mpnet", "random"]
    model_types = [
        # "baseline",
        # "onehot",
        # "peb",
        # "word_bias",
        # "bias",
        # "embedding",
        # "word_embedding",
        "transformer_user_id",
    ]
    fold_nums = 10

    append_annotator_ids = (
        True  # If true, use UserID model, else use standard transfromer
    )
    batch_size = 10
    epochs = 3
    lr_rate = 5e-5

    use_cuda = True

    for (datamodule_cls, embeddings_type, stratify_by,) in product(
        datamodule_clses,
        embedding_types,
        stratify_by_options,
    ):

        seed_everything(seed=22)
        data_module = datamodule_cls(
            embeddings_type=embeddings_type,
            normalize=regression,
            batch_size=batch_size,
            stratify_folds_by=stratify_by,
            major_voting=True,
        )
        data_module.prepare_data()
        data_module.setup()
        data_module.compute_word_stats(
            min_word_count=200,
            min_std=0.0,
            words_per_text=100,
        )

        for model_type, fold_num in product(model_types, range(fold_nums)):

            data_module._recompute_stats_for_fold(fold_num)

            hparams = {
                "dataset": type(data_module).__name__,
                "model_type": model_type,
                "embeddings_type": embeddings_type,
                "fold_num": fold_num,
                "regression": regression,
                "stratify_by": stratify_by,
                "append_annotator_ids": append_annotator_ids,
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
                word_num=data_module.words_number + 2,
                annotator_num=data_module.annotators_number + 2,
                dp=0.0,
                dp_emb=0.25,
                embedding_dim=50,
                hidden_dim=100,
                bias_vector_length=len(data_module.class_dims),
                append_annotator_ids=append_annotator_ids,
                huggingface_model_name=TRANSFORMER_MODEL_STRINGS[embeddings_type],
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
                custom_callbacks=[
                    # callbacks.SaveOutputsWandb(
                    #     save_name="wandb_outputs.csv", save_text=True
                    # ),
                    callbacks.SaveOutputsLocal(
                        save_dir="wiki_experiments_outputs",
                        fold_num=fold_num,
                        experiment="wiki_test",
                    ),
                ],
            )

            logger.experiment.finish()
