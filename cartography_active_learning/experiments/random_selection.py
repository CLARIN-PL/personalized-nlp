import numpy as np
import pandas as pd

from settings import LOGS_DIR
import pytorch_lightning.callbacks as callbacks
from pytorch_lightning import loggers as pl_loggers
from cartography_active_learning.datasets import CartographyDataModule, RegressorDataModule
from cartography_active_learning.utils import get_seed

from cartography_active_learning.learning import train_model_cartography, train_regressor
from cartography_active_learning.utils.cartography import get_cartography

def main():
    
    PROJECT_NAME = 'CartographyActiveLearningBugFix'
    SPLIT_SIZES = [0.1, 0.15, 0.2]
    METRIC = "random"
    SEED_SIZES = [0.001, 0.05, 0.1, 0.15, 0.2]
    MODELS = ['baseline']
    MODEL_KWARGS = {"embedding_dim": 50,
            "dp_emb": 0.25,
            "dp": 0.0,
            "hidden_dim": 100}
    MAX_EPOCHS = 100
    STEP_SIZES = [0.1, 0.01]
    TEST_FOLDS = [i for i in range(10)]
    
    for split_size in SPLIT_SIZES:
        for seed_size in SEED_SIZES:
            for step_size in STEP_SIZES:
                for model_type in MODELS:
                    for test_fold in TEST_FOLDS:
                        val_fold = (test_fold + 1) % 10
                        
                        # read the data
                        data_original = pd.read_csv('/home/konradkaranowski/storage/personalized-nlp/storage/data/wiki_data/toxicity_annotations_420balanced_folds.csv')
                        texts = pd.read_csv('/home/konradkaranowski/storage/personalized-nlp/storage/data/wiki_data/toxicity_annotated_comments_processed.csv')
                        data = get_seed(data_original, test_fold=test_fold, val_fold=val_fold, perc=seed_size)

                        datamodule = CartographyDataModule(
                            annotations=data,
                            data=texts,
                            test_fold=test_fold,
                            val_fold=val_fold
                        )
                        while datamodule.train_pct_used < 1.0:
                            
                            hparams = {
                                    "dataset": type(datamodule).__name__,
                                    "train_size": len(datamodule.train_dataloader().dataset),
                                    "pct": datamodule.train_pct_used,
                                    "unused_size": len(datamodule.unused_train),
                                    "model_type": model_type,
                                    "model": "train_model",
                                    "seed_size": seed_size,
                                    "step_size": step_size,
                                    "test_fold": test_fold,
                                    "metric": METRIC,
                                    "split_size": split_size
                                }

                            logger = pl_loggers.WandbLogger(
                                    save_dir=str(LOGS_DIR),
                                    config=hparams,
                                    project=PROJECT_NAME,
                                    log_model=False,
                                )
                            
                            _ = train_model_cartography(
                                model_type, 
                                datamodule, 
                                model_kwargs=MODEL_KWARGS, 
                                logger=logger, 
                                epochs=MAX_EPOCHS, 
                                custom_callbacks=[
                                    callbacks.EarlyStopping(monitor="valid_loss", mode="min", patience=3)
                                ])
                           
                            train = datamodule.unused_train
                            
                            train['random'] = np.random.sample(size=len(train))
                            datamodule.add_top_k(train, metric='random', amount=int(datamodule.train_size * step_size), ascending=False)
        
if __name__ == '__main__':
    main()

