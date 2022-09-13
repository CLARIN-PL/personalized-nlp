import numpy as np
import pandas as pd

from cartography_active_learning.datasets import CartographyDataModule
from cartography_active_learning.utils import get_seed


def main():
    
    test_fold, val_fold = 0, 1
    data = pd.read_csv('/home/konradkaranowski/storage/personalized-nlp/storage/data/wiki_data/toxicity_annotations_420balanced_folds.csv')
    texts = pd.read_csv('/home/konradkaranowski/storage/personalized-nlp/storage/data/wiki_data/toxicity_annotated_comments_processed.csv')
    data = get_seed(data, test_fold=test_fold, val_fold=val_fold, perc=0.2)

    datamodule = CartographyDataModule(
        annotations=data,
        data=texts,
        test_fold=test_fold,
        val_fold=val_fold
    )
    datamodule.prepare_data()
    datamodule.setup()
    while datamodule.train_pct_used < 1.0:
        tr = datamodule.train_dataloader()
        vl = datamodule.val_dataloader()
        ts = datamodule.test_dataloader()
        print(f'PCT: {round(datamodule.train_pct_used, 2)} Train: {len(tr.dataset)} Val: {len(vl.dataset)} Test: {len(ts.dataset)}')
        unused = datamodule.unused_train
        unused['x'] = np.random.sample(size=len(unused))
        datamodule.add_top_k(unused, metric='x', amount=int(datamodule.train_size * 0.1))
        
    # for fold in folds:
    #     for seed_size in seed_sizes:
            # for step_size in step_sizes:
            #     datamodule = CartographyDataModule()
            #     while datamodule.pct_used() < maximal_pct_used:            
            #         training_dynamics = train_model(model, datamodule)
            #         cartography = calculate_cartography(training_dynamics)
                    
            #         regressor_datamodule = RegressorDataModule(cartography)
            #         regressor = Regressor()
            #         train_regressor(regressor, regressor_datamodule)
            #         predictions = predict_regressor()
            #         datamodule.add_top_k(predictions, metric, to_add)
    pass

if __name__ == '__main__':
    main()

