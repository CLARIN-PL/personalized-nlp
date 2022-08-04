from personalized_nlp.utils.experiments import product_kwargs
    
  
GRIDS = {
    'grid_nice': {
        'model_kwargs_list': product_kwargs(
            {
                "embedding_dim": [50],
                "dp_emb": [0.25],
                "dp": [0.0],
                "hidden_dim": [100],
            }
        ),
        'flow_kwargs_list': product_kwargs(
            {
                "hidden_features": [2, 4, 6, 8],
                "num_layers":  [1, 2, 3, 4, 5],
                "num_blocks_per_layer":  [1, 2, 3, 4],
                "dropout_probability": [0.0, 0.1, 0.3, 0.4],
                "batch_norm_within_layers": [True, False],
                "batch_norm_between_layers":  [True, False],
            }
        ),
        'trainer_kwargs_list': product_kwargs(
            {
                "epochs": [500],
                "lr_rate": [1e-4, 1e-5],
                "use_cuda": [True],
                "flow_type": [ 
                    'nice'
                ],
                "model_type": [
                    'flow_baseline', 
                    'flow_onehot',
                    'flow_peb'
                ]
            }
        )
    },
    'grid_real_nvp': {
        'model_kwargs_list': product_kwargs(
            {
                "embedding_dim": [50],
                "dp_emb": [0.25],
                "dp": [0.0],
                "hidden_dim": [100],
            }
        ),
        'flow_kwargs_list': product_kwargs(
            {
                "hidden_features": [2, 4, 6, 8],
                "num_layers":  [1, 2, 3, 4, 5],
                "num_blocks_per_layer":  [1, 2, 3, 4],
                "dropout_probability": [0.0, 0.1, 0.3, 0.4],
                "batch_norm_within_layers": [True, False],
                "batch_norm_between_layers":  [True, False],
            }
        ),
        'trainer_kwargs_list': product_kwargs(
            {
                "epochs": [500],
                "lr_rate": [1e-4, 1e-5],
                "use_cuda": [True],
                "flow_type": [ 
                    'real_nvp'
                ],
                "model_type": [
                    'flow_baseline', 
                    'flow_onehot',
                    'flow_peb'
                ]
            }
        )
    },
    'grid_maf': {
        'model_kwargs_list': product_kwargs(
            {
                "embedding_dim": [50],
                "dp_emb": [0.25],
                "dp": [0.0],
                "hidden_dim": [50]
            }
        ),
        'flow_kwargs_list': product_kwargs(
            {
                "hidden_features": [2, 4, 6, 8],
                "num_layers": [1, 2, 3, 4, 5],
                "num_blocks_per_layer": [1, 2, 3, 4],
                "batch_norm_within_layers": [True, False],
                "batch_norm_between_layers": [True, False]
            }
        ),
        'trainer_kwargs_list': product_kwargs(
            {
                "epochs": [500],
                "lr_rate": [1e-4, 1e-5],
                "use_cuda": [True],
                "flow_type": [ 
                    'maf'
                ],
                "model_type": [
                    'flow_baseline',
                    'flow_peb',
                    'flow_onehot'
                ]
            }
        )         
    },
    'grid_bias_maf': {
        'model_kwargs_list': product_kwargs(
            {
                "embedding_dim": [50],
                "dp_emb": [0.25],
                "dp": [0.0],
                "hidden_dim": [128, 256, 512, 768],
                "output_dim": [128, 256, 512, 768]
            }
        ),
        'flow_kwargs_list': product_kwargs(
            {
                "hidden_features": [2, 4, 6, 8],
                "num_layers": [1, 2, 3, 4, 5],
                "num_blocks_per_layer": [1, 2, 3, 4],
                "batch_norm_within_layers": [True, False],
                "batch_norm_between_layers": [True, False]
            }
        ),
        'trainer_kwargs_list': product_kwargs(
            {
                "epochs": [500],
                "lr_rate": [1e-4, 1e-5],
                "use_cuda": [True],
                "flow_type": [ 
                    'maf'
                ],
                "model_type": [
                    'flow_bias'
                ]
            }
        ) 
    },
    'grid_bias_nvp_nice':{
        'model_kwargs_list': product_kwargs(
            {
                "embedding_dim": [50],
                "dp_emb": [0.25],
                "dp": [0.0],
                "hidden_dim": [128, 256, 512, 768],
                "output_dim": [128, 256, 512, 768]
            }
        ),
        'flow_kwargs_list': product_kwargs(
            {
                "hidden_features": [2, 4, 6, 8],
                "num_layers":  [1, 2, 3, 4, 5],
                "num_blocks_per_layer":  [1, 2, 3, 4],
                "dropout_probability": [0.0, 0.1, 0.3, 0.4],
                "batch_norm_within_layers": [True, False],
                "batch_norm_between_layers":  [True, False],
            }
        ),
        'trainer_kwargs_list': product_kwargs(
            {
                "epochs": [500],
                "lr_rate": [1e-4, 1e-5],
                "use_cuda": [True],
                "flow_type": [ 
                    'real_nvp',
                    'nice'
                ],
                "model_type": [
                    'flow_bias'
                ]
            }
        )        
    }
}  
    