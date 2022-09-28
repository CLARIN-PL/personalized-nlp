import os

import personalized_nlp.utils.callbacks as callbacks
from personalized_nlp.learning.train import train_test
from pytorch_lightning.callbacks import EarlyStopping
from settings import TRANSFORMER_MODEL_STRINGS


def finetune_datamodule_embeddings(
    original_datamodule,
    batch_size: int = 32,  # 20
    epochs=50,  # 3
    lr_rate=0.00005,
    use_cuda=True,
):

    embeddings_type = original_datamodule.embeddings_type
    stratify_folds_by = original_datamodule.stratify_folds_by
    data_dir = original_datamodule.data_dir
    fold_num = original_datamodule.test_fold

    embeddings_path = (
        f"{data_dir}/embeddings/{embeddings_type}_{fold_num}_{stratify_folds_by}.p"
    )

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
        "max_length": 512,  # 128
    }

    train_test(
        datamodule,
        model_kwargs=model_kwargs,
        model_type="transformer_user_id",
        epochs=epochs,
        lr=lr_rate,
        regression=regression,
        use_cuda=use_cuda,
        custom_callbacks=[
            callbacks.SaveEmbeddingCallback(
                datamodule=datamodule,
                save_path=embeddings_path,
            ),
            EarlyStopping(
                # monitor="valid_accuracy_sentiment",
                monitor="valid_macro_f1_sentiment",
                mode="max",
                patience=10)
        ],
    )
