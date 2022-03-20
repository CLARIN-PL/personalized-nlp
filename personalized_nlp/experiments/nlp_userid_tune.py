import os
import torch
import random
import numpy as np
from itertools import product
from personalized_nlp.learning.train import train_test
from personalized_nlp.models import models as models_dict
from personalized_nlp.settings import LOGS_DIR
from pytorch_lightning import loggers as pl_loggers
from personalized_nlp.datasets.emotions_perspective.emotions_perspectives import EmotionsPerspectiveDataModule
from transformers import get_scheduler
from torch.optim import AdamW
from datasets import load_metric

def seed_everything():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    regression = True
    datamodule_cls = EmotionsPerspectiveDataModule
    embedding_types = ['roberta']
    model_types = ['userid']
    wandb_project_name = 'nlpersTed'
    limit_past_annotations_list = [None] # range(20)
    fold_nums = 10
    min_annotations_per_text = 2
    
    min_word_counts = [50]
    words_per_texts = [128]
    
    batch_size = 16
    dp_embs = [0.25]
    embedding_dims = [50]
    epochs = 2
    lr_rate = 0.1
    w_decay = 0.01
    
    use_cuda = True

    for (min_word_count, words_per_text, embeddings_type, limit_past_annotations) in product(
        min_word_counts, words_per_texts, embedding_types, limit_past_annotations_list
    ):

        seed_everything()
        data_module = datamodule_cls(embeddings_type=embeddings_type, normalize=regression,
                                     batch_size=batch_size, past_annotations_limit=limit_past_annotations)
        data_module.prepare_data()
        data_module.setup()
        data_module.compute_word_stats(
            min_word_count=min_word_count,
            min_std=0.0,
            words_per_text=words_per_text,
        )

        train_dataloader = data_module.train_dataloader()

        val_dataloader = data_module.val_dataloader()

        test_dataloader = data_module.test_dataloader()

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
                bias_vector_length=len(data_module.class_dims)
            )

            optimizer = AdamW(model.parameters(), lr=lr_rate)
            num_training_steps = epochs * len(train_dataloader)
            lr_scheduler = get_scheduler(name="linear", optimizer=optimizer,
                                         num_warmup_steps=0,
                                         num_training_steps=num_training_steps)
            
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            model.to(device)

            model.train()
            for epoch in range(epochs):
                for batch in train_dataloader:
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            
            metric = load_metric("f1")
            model.eval()
            for batch in val_dataloader:
                with torch.no_grad():
                    outputs = model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])

            metric.compute()

            logger.experiment.finish()
