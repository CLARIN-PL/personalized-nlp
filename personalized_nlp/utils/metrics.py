
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, r2_score
from collections import defaultdict

def evaluate_classification(test_predictions, true_labels, class_dims):
    dims_results = {}
    for cls_idx in range(len(class_dims)):
        start_idx =  sum(class_dims[:cls_idx])
        end_idx =  start_idx + class_dims[cls_idx]
        preds = np.argmax(test_predictions[:, start_idx:end_idx], axis=1)
        
        if true_labels.ndim > 1:
            targets = true_labels[:, cls_idx]
        else:
            targets = true_labels

        dims_results[cls_idx] = classification_report(targets, preds, output_dict=True)
    
    return dims_results

def evaluate_regression(test_predictions, true_labels):
    if true_labels.ndim > 1:
        losses = [r2_score(true_labels[:, i], test_predictions[:, i]) for i in range(test_predictions.shape[1])]
    else:
        losses = [r2_score(true_labels, test_predictions)]

    return losses


def get_result_dataframe(results, regression=False, class_names=None):
    result_dict = defaultdict(list)

    for experiment in results:
        hparams = experiment[0]
        metrics = experiment[1]
        
        for hparam_key, hparam_value in hparams.items():
            result_dict[hparam_key].append(hparam_value)
        
        if regression:
            for class_idx, r_squared in enumerate(metrics['r_squared']):
                if class_names is not None:
                    class_name = class_names[class_idx]
                else:
                    class_name = str(class_idx)

                result_dict['r_squared_' + class_name].append(r_squared) 
        else:
            for class_idx, values in metrics.items():
                if not isinstance(class_idx, int):
                    continue

                if class_names is not None:
                    class_name = class_names[class_idx]
                else:
                    class_name = str(class_idx)

                f1_score = values['1']['f1-score']
                macro_f1_score = values['macro avg']['f1-score']    
                accuracy = values['accuracy']
                result_dict['f1_score_' + class_name].append(f1_score)
                result_dict['macro_f1_' + class_name].append(macro_f1_score)
                result_dict['accuracy_' + class_name].append(accuracy)

        training_time = metrics['training_time']
        testing_time = metrics['testing_time']

        result_dict['training_time'].append(training_time)
        result_dict['testing_time'].append(testing_time)

    return pd.DataFrame(result_dict)