import os
from itertools import product

from personalized_nlp.learning.train import train_test
from personalized_nlp.models import models as models_dict
from personalized_nlp.settings import LOGS_DIR
from personalized_nlp.datasets.measuring_hate_speech.measuring_hate_speech import MeasuringHateSpeechDataModule
from personalized_nlp.utils import seed_everything
from personalized_nlp.utils.callbacks.outputs import SaveOutputsWandb, SaveOutputsLocal

from pytorch_lightning import loggers as pl_loggers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_entity_name = 'persemo'
    wandb_project_name = "MeasuringHateSpeech"  # "MeasuringHateSpeechBaselineMajorityVoting"

    regression = True
    datamodule_clses = [MeasuringHateSpeechDataModule]
    stratify_by_options = [
        None,
        "users",
        "texts",
    ][-1:]
    embedding_types = ["xlmr", "bert", "deberta", "mpnet", "random"][:1]
    model_types = [
        # "onehot",
        "baseline",
        "peb",
        "embedding",
        # "word_bias",
        # "bias",
        # "word_embedding",
        "transformer_user_id",
    ][-1:]
    fold_nums = 10

    majority_vote = False
    append_annotator_ids = (
        True  # If true, use UserID model, else use standard transfromer
    )
    batch_size = 30  # 10 for transformer 4096 otherwise
    epochs = 3  # 3 for transformer 180 otherwise
    lr_rate = 1e-5  # 1e-5 for transformer 0.0002 otherwise

    use_cuda = True

    for (
            datamodule_cls,
            embeddings_type,
            stratify_by,
    ) in product(
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
            major_voting=majority_vote,
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
                'majority_vote': majority_vote
            }

            logger = pl_loggers.WandbLogger(
                save_dir=LOGS_DIR,
                config=hparams,
                entity=wandb_entity_name,
                project=wandb_project_name,
                log_model=False,
            )

            output_dim = (len(data_module.class_dims)
                          if regression else sum(data_module.class_dims))
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
                    # SaveOutputsWandb(save_name="wandb_outputs.csv",
                    #                  save_text=True),
                    SaveOutputsLocal(
                        save_dir=
                        f"lrec_{type(data_module).__name__}_{model_type}",
                        # fold_num=fold_num,
                        experiment="measuring_hate_speech",
                        **hparams),
                ],
            )

            logger.experiment.finish()