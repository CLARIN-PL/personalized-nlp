import os

import personalized_nlp.utils.callbacks as callbacks
from personalized_nlp.learning.train import train_test
from settings import TRANSFORMER_MODEL_STRINGS
from pytorch_lightning import loggers as pl_loggers
from settings import LOGS_DIR


def finetune_datamodule_embeddings(
    original_datamodule,
    batch_size: int = 16,
    epochs=4,
    lr=8e-6,
    use_cuda=True,
    wandb_project_name=None,
):

    embeddings_type = original_datamodule.embeddings_type
    stratify_folds_by = original_datamodule.stratify_folds_by
    data_dir = original_datamodule.data_dir
    fold_num = original_datamodule.test_fold

    embeddings_path = f"{data_dir}/embeddings/finetuned_{embeddings_type}_{fold_num}_{stratify_folds_by}.p"

    if os.path.exists(embeddings_path):
        return

    datamodule_cls = type(original_datamodule)
    init_args = dict(original_datamodule._init_args)
    init_args["major_voting"] = True
    init_args["batch_size"] = batch_size
    init_args["use_finetuned_embeddings"] = False

    datamodule = datamodule_cls(**init_args)

    regression = datamodule.regression
    model_kwargs = {
        "huggingface_model_name": TRANSFORMER_MODEL_STRINGS[embeddings_type],
        "max_length": 128,
    }

    if wandb_project_name is not None:
        hparams = {
            "dataset": type(datamodule).__name__,
            **init_args,
        }

        logger = pl_loggers.WandbLogger(
            save_dir=str(LOGS_DIR),
            config=hparams,
            project=wandb_project_name,
            log_model=False,
        )
    else:
        logger = None

    train_test(
        datamodule,
        model_kwargs=model_kwargs,
        model_type="transformer_user_id",
        epochs=epochs,
        lr=lr,
        regression=regression,
        use_cuda=use_cuda,
        custom_callbacks=[
            callbacks.SaveEmbeddingCallback(
                datamodule=datamodule,
                save_path=embeddings_path,
            )
        ],
        logger=logger,
    )
    
    if logger is not None:
        logger.experiment.finish()
