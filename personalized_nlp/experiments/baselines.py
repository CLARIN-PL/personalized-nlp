import numpy as np

from personalized_nlp.datasets.wiki.aggression import AggressionDataModule

from sklearn.model_selection import cross_val_score
from sklearn import svm

if __name__ == '__main__':
    data_module = AggressionDataModule(embeddings_type='xlmr')
    data_module.prepare_data()
    data_module.setup()

    annotations = data_module.annotations.copy()
    embeddings = data_module.text_embeddings.to('cpu').numpy()

    annotations['text_id'] = annotations['text_id'].apply(
        lambda r_id: data_module.text_id_idx_dict[r_id])
    annotations['annotator_id'] = annotations['annotator_id'].apply(
        lambda w_id: data_module.annotator_id_idx_dict[w_id])

    X = np.vstack([embeddings[i] for i in annotations['text_id'].values])
    y = annotations[data_module.annotation_column].values

    clf = svm.SVC(kernel='linear', C=1, random_state=42)
    scores = cross_val_score(clf, X, y, cv=3)

    print(scores)
