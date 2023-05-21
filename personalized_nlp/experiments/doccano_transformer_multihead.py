import os
from itertools import product

from personalized_nlp.datasets.doccano.doccano import DoccanoDataModule

from personalized_nlp.learning.train import train_test
from settings import LOGS_DIR
from personalized_nlp.utils import seed_everything
from personalized_nlp.utils.experiments import product_kwargs
from settings import TRANSFORMER_MODEL_STRINGS
from personalized_nlp.utils.callbacks.outputs import SaveOutputsLocal, SaveOutputsWandb
from pytorch_lightning import loggers as pl_loggers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = "DoccanoTransformerMultiHead"
    datamodule_cls = DoccanoDataModule

    datamodule_kwargs_list = product_kwargs(
        {
            "regression": [True],
            "embeddings_type": ["labse"],
            "limit_past_annotations_list": [None],
            "stratify_folds_by": ["users", "texts"][1:],
            "folds_num": [5],
            "batch_size": [16],
            "test_fold": list(range(5)),
            "use_finetuned_embeddings": [False],
            "major_voting": [False],
            "empty_annotations_strategy": ["drop"],
            "min_annotations_per_text": [20],
        }
    )
    model_kwargs_list = product_kwargs(
        {
            "huggingface_model_name": [TRANSFORMER_MODEL_STRINGS["labse"]],
            "append_annotator_ids": [False, True],
            "use_cuda": [True],
        }
    )
    trainer_kwargs_list = product_kwargs(
        {
            "epochs": [6],
            "lr": [1e-5],
            "regression": [True],
            "use_cuda": [True],
            "model_type": ["transformer_multihead", "transformer_user_id"],
        }
    )

    for datamodule_kwargs in datamodule_kwargs_list:
        seed_everything()
        data_module = datamodule_cls(**datamodule_kwargs)

        for model_kwargs, trainer_kwargs in product(
            model_kwargs_list,
            trainer_kwargs_list,
        ):
            if (
                trainer_kwargs["model_type"] == "transformer_multihead"
                and model_kwargs["append_annotator_ids"]
            ):
                continue

            model_kwargs["huggingface_model_name"] = TRANSFORMER_MODEL_STRINGS[
                datamodule_kwargs["embeddings_type"]
            ]
            hparams = {
                "dataset": type(data_module).__name__,
                **datamodule_kwargs,
                **model_kwargs,
                **trainer_kwargs,
            }

            try:
                logger = pl_loggers.WandbLogger(
                    save_dir=str(LOGS_DIR),
                    config=hparams,
                    project=wandb_project_name,
                    log_model=False,
                )
            except:
                continue

            model_type = trainer_kwargs["model_type"]
            train_test(
                datamodule=data_module,
                model_kwargs=model_kwargs,
                logger=logger,
                # custom_callbacks=[
                #     SaveOutputsWandb(f"doccano_outputs_{model_type}", save_text=False)
                # ],
                **trainer_kwargs,
            )

            logger.experiment.finish()
