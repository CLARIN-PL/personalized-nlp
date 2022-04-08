import os
from itertools import product
import torch

from personalized_nlp.learning.train import train_test
from personalized_nlp.models import models as models_dict
from personalized_nlp.settings import LOGS_DIR
from personalized_nlp.datasets.emotions_perspective.emotions_perspectives import EmotionsPerspectiveDataModule
from personalized_nlp.utils import seed_everything
from pytorch_lightning import loggers as pl_loggers

from personalized_nlp.utils.callbacks.optimizer import SetWeightDecay
from personalized_nlp.utils.callbacks.transformer_lr_scheduler import TransformerLrScheduler

torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = 'test_on_epoch_hook_2'

    regression = True
    datamodule_cls = EmotionsPerspectiveDataModule
    embedding_types = ['roberta']
    model_types = ['hubi_med']
    limit_past_annotations_list = [None] 
    fold_nums = 10
    
    min_word_counts = [50]
    words_per_texts = [15]
    
    batch_size = 16
    dp_embs = [0.25]
    embedding_dims = [50]
    epochs = 20
    lr_rate = 3e-5
    weight_decay = 1e-6
    # set nr_frozen_epochs = 0 to finetuning from scratch, = epochs to frozen the whole
    nr_frozen_epochs = 0

    user_folding = True
    use_cuda = True
    # frozen=False by default for finetuning, set frozen=True for not finetuning
    frozen = True

    for (min_word_count, words_per_text, embeddings_type, limit_past_annotations) in product(
        min_word_counts, words_per_texts, embedding_types, limit_past_annotations_list
    ):

        seed_everything()
        data_module = datamodule_cls(
            embeddings_type=embeddings_type, normalize=regression, batch_size=batch_size,
            past_annotations_limit=limit_past_annotations
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
                "weight_decay": weight_decay,
                'nr_frozen_epochs': nr_frozen_epochs
            }

            logger = pl_loggers.WandbLogger(
                save_dir=LOGS_DIR,
                config=hparams,
                project=wandb_project_name,
                log_model=False,
            )

            output_dim = len(data_module.class_dims) if regression else sum(data_module.class_dims)
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
                nr_frozen_epochs=nr_frozen_epochs,
                embedding_type=embeddings_type,
                frozen = frozen
            )

            custom_callbacks = [SetWeightDecay(lr=lr_rate, weight_decay=weight_decay)]
            if frozen==False:
                custom_callbacks += [TransformerLrScheduler(warmup_proportion=0.1)]

            train_test(
                data_module,
                model,
                epochs=epochs,
                lr=lr_rate,
                weight_decay=weight_decay,
                regression=regression,
                use_cuda=use_cuda,
                logger=logger,
                test_fold=fold_num,
                nr_frozen_epochs=nr_frozen_epochs,
                custom_callbacks=custom_callbacks
            )

            logger.experiment.finish()
            