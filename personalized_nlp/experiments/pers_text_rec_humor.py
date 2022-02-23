import os
from itertools import product
from functools import partial
import numpy as np

from personalized_nlp.learning.train import train_test
from personalized_nlp.models import models as models_dict
from personalized_nlp.settings import LOGS_DIR
from personalized_nlp.datasets.humor.humor import HumorDataModule
# from personalized_nlp.datasets.wiki.toxicity import ToxicityDataModule
# from personalized_nlp.datasets.wiki.attack import AttackDataModule
# from personalized_nlp.datasets.wiki.aggression import AggressionDataModule
from personalized_nlp.utils import seed_everything
from personalized_nlp.utils.callbacks.outputs import SaveOutputsCallback
from personalized_nlp.utils.pers_text_rec import (
    assign_annotations, 
    identity,
    get_var_ratio_annotation_count_weighted, 
    get_weighted_conformity_annotation_count_weighted, 
    get_text_controversy_annotation_count_weighted, 
    get_entropy,
    # nowe
    get_var_ratio,
    get_conformity,
    get_weighted_conformity,
    get_avg_max_conformity,
    get_avg_min_conformity,
    get_max_weighted_conformity,
    get_max_min_conformity,
    get_max_max_conformity,
    get_min_weighted_conformity,
    get_min_max_conformity,
    get_min_min_conformity,
    get_mean_conformity_annotation_count_weighted,
    ) 
from personalized_nlp.utils.measures import set_first_assignemnt, measure_annotation_distance, random_assignment
from pytorch_lightning import loggers as pl_loggers

# os.environ["CUDA_VISIBLE_DEVICES"] = "99"  # "1"
os.environ["WANDB_START_METHOD"] = "thread"

# HYPERPARAMETERS FOR PERS TEXT REC MECHANISM
MAX_ANNOTATIONS_PER_USER = 100
MAX_USER_ANNOTATIONS_ORDER_DEV_TEST = np.arange(1, 15, 1)
# MAX_USER_ANNOTATIONS_ORDER_TRAIN = np.arange(0, 15, 1)
COLUMN_NAME = 'aggression'
PAST_PRESENT_SPLIT = True

if __name__ == "__main__":
    regression = False
    datamodule_clses = [
        # AggressionDataModule
        HumorDataModule
    ]
    embedding_types = [
        'xlmr'
    ]  # ['random', 'cbow', 'skipgram', 'labse', 'mpnet', 'xlmr', 'deberta', 'bert']
    model_types = [
        #'embedding',
        'baseline', 
        'peb', 
        #'bias', 
    ]

    all_annotation_rules = [
        {
            'first_rule': {
                'name': 'random_assignment',
                'rule': random_assignment
            },
            'next_rule': {
                'name': 'identity',
                'rule': identity
            }
        },
        {
            'first_rule': {
                'name': 'get_var_ratio_annotation_count_weighted',
                'rule': get_var_ratio_annotation_count_weighted
            },
            'next_rule': {
                'name': 'identity',
                'rule': identity
            }
        },
        {
            'first_rule': {
                'name': 'get_weighted_conformity_annotation_count_weighted',
                'rule': get_weighted_conformity_annotation_count_weighted
            },
            'next_rule': {
                'name': 'identity',
                'rule': identity
            }
        },
        {
            'first_rule': {
                'name': 'get_text_controversy_annotation_count_weighted',
                'rule': get_text_controversy_annotation_count_weighted
            },
            'next_rule': {
                'name': 'identity',
                'rule': identity
            }
        },
        {
            'first_rule': {
                'name': 'random_first_assignment',
                'rule': set_first_assignemnt
            },
            'next_rule': {
                'name': 'measure_annotation_distance',
                'rule': partial(measure_annotation_distance, center_of_weight_method=np.mean)
            }
        },
        # {
        #     'first_rule': {
        #         'name': 'get_var_ratio',
        #         'rule': get_var_ratio
        #     },
        #     'next_rule': {
        #         'name': 'identity',
        #         'rule': identity
        #     }
        # },
        # {
        #     'first_rule': {
        #         'name': 'get_conformity',
        #         'rule': get_conformity
        #     },
        #     'next_rule': {
        #         'name': 'identity',
        #         'rule': identity
        #     }
        # },
        # {
        #     'first_rule': {
        #         'name': 'get_weighted_conformity',
        #         'rule': get_weighted_conformity
        #     },
        #     'next_rule': {
        #         'name': 'identity',
        #         'rule': identity
        #     }
        # },
        # {
        #     'first_rule': {
        #         'name': 'get_avg_max_conformity',
        #         'rule': get_avg_max_conformity
        #     },
        #     'next_rule': {
        #         'name': 'identity',
        #         'rule': identity
        #     }
        # },
        # {
        #     'first_rule': {
        #         'name': 'get_avg_min_conformity',
        #         'rule': get_avg_min_conformity
        #     },
        #     'next_rule': {
        #         'name': 'identity',
        #         'rule': identity
        #     }
        # },
        # {
        #     'first_rule': {
        #         'name': 'get_max_weighted_conformity',
        #         'rule': get_max_weighted_conformity
        #     },
        #     'next_rule': {
        #         'name': 'identity',
        #         'rule': identity
        #     }
        # },
        # {
        #     'first_rule': {
        #         'name': 'get_max_min_conformity',
        #         'rule': get_max_min_conformity
        #     },
        #     'next_rule': {
        #         'name': 'identity',
        #         'rule': identity
        #     }
        # },
        # {
        #     'first_rule': {
        #         'name': 'get_max_max_conformity',
        #         'rule': get_max_max_conformity
        #     },
        #     'next_rule': {
        #         'name': 'identity',
        #         'rule': identity
        #     }
        # },
        # {
        #     'first_rule': {
        #         'name': 'get_min_weighted_conformity',
        #         'rule': get_min_weighted_conformity
        #     },
        #     'next_rule': {
        #         'name': 'identity',
        #         'rule': identity
        #     }
        # },
        # {
        #     'first_rule': {
        #         'name': 'get_min_max_conformity',
        #         'rule': get_min_max_conformity
        #     },
        #     'next_rule': {
        #         'name': 'identity',
        #         'rule': identity
        #     }
        # },
        # {
        #     'first_rule': {
        #         'name': 'get_min_min_conformity',
        #         'rule': get_min_min_conformity
        #     },
        #     'next_rule': {
        #         'name': 'identity',
        #         'rule': identity
        #     }
        # },
        # {
        #     'first_rule': {
        #         'name': 'get_mean_conformity_annotation_count_weighted',
        #         'rule': get_mean_conformity_annotation_count_weighted
        #     },
        #     'next_rule': {
        #         'name': 'identity',
        #         'rule': identity
        #     }
        # }
    ]

    wandb_entity_name = 'persemo'
    # wandb_project_name = 'PersTextRecWikiAggressionNoPastPresentFixed'
    wandb_project_name = 'PersTextRecWikiAggressionPastPresent'
    fold_nums = 10

    min_word_counts = [50]
    words_per_texts = [15]

    batch_size = 3000
    dp_embs = [0.25]
    embedding_dims = [50]
    epochs = 20
    lr_rate = 0.008  # 0.008

    use_cuda = True

    for (datamodule_cls, min_word_count, words_per_text, embeddings_type,
         annotation_rules) in product(
             datamodule_clses, min_word_counts, words_per_texts,
             embedding_types, all_annotation_rules):
             
        # IMPORTANT create assign annotations function
        assign_func = partial(
            assign_annotations,
            column_name=COLUMN_NAME,
            first_annotation_rule=annotation_rules['first_rule']['rule'],
            next_annotations_rule=annotation_rules['next_rule']['rule'],
            max_annotations_per_user=MAX_ANNOTATIONS_PER_USER
        )
        # create datamodule
        data_module = datamodule_cls(
            embeddings_type=embeddings_type,
            normalize=regression,
            batch_size=batch_size,
            past_annotations_limit=None,
            # PASS THIS FUNCTION INTO THE DATAMODULE
            assign_annotations_function=assign_func)
        data_module.prepare_data()

        # DECIDE SPLIT TACTIC!
        data_module.past_present_split = PAST_PRESENT_SPLIT

        data_module.setup()
        data_module.compute_word_stats(
            min_word_count=min_word_count,
            min_std=0.0,
            words_per_text=words_per_text,
        )


        for model_type, embedding_dim, dp_emb, fold_num in product(
                model_types, embedding_dims, dp_embs, range(fold_nums)):
            # ITERATE OVER NUMBER OF USERS IN ONE MODEL
            for max_user_annotation_order_dev_test in MAX_USER_ANNOTATIONS_ORDER_DEV_TEST:
                # IMPORTANT!!!!!!!!!!!!
                # REMEMBER TO SET max_user_annotation_order PROPERTY IN DATAMODULE!!! 
                data_module.max_user_annotation_order_dev_test = max_user_annotation_order_dev_test
                data_module.max_user_annotation_order_train = max_user_annotation_order_dev_test
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
                            # pers text rec
                            "max_user_annotation_order_dev_test": data_module.max_user_annotation_order_dev_test,
                            "max_user_annotation_order_train": data_module.max_user_annotation_order_train,
                            "first_annotation_rule": annotation_rules['first_rule']['name'],
                            "next_annotation_rules": annotation_rules['next_rule']['name'],
                            "past_present_split": data_module.past_present_split 
                        }

                logger = pl_loggers.WandbLogger(
                    save_dir=LOGS_DIR,
                    config=hparams,
                    project=wandb_project_name,
                    entity=wandb_entity_name,
                    log_model=False,
                )

                output_dim = len(

                    data_module.class_dims) if regression else sum(
                        data_module.class_dims)
                text_embedding_dim = data_module.text_embedding_dim
                model_cls = models_dict[model_type]

                model = model_cls(output_dim=output_dim,
                                    text_embedding_dim=text_embedding_dim,
                                    word_num=data_module.words_number,
                                    annotator_num=data_module.annotators_number,
                                    dp=0.0,
                                    dp_emb=dp_emb,
                                    embedding_dim=embedding_dim,
                                    hidden_dim=100,
                                    bias_vector_length=len(
                                        data_module.class_dims))

                train_test(
                    data_module,
                    model,
                    epochs=epochs,
                    lr=lr_rate,
                    regression=regression,
                    use_cuda=use_cuda,
                    logger=logger,
                    test_fold=fold_num,
                    # custom_callbacks=[
                    #     SaveOutputsCallback(save_dir=os.path.join(data_module.data_dir, 'outputs'))
                    # ]
                )

                logger.experiment.finish()          
      