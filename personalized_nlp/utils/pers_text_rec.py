from cgitb import text
from pstats import Stats
import string
from scipy.stats import rankdata, mode, entropy
from typing import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def rank(rows):
    # len(rows) - rankdata(rows, method='ordinal')
    rows["user_annotation_order"] = len(rows) - rankdata(rows['measure_value'].values, method="ordinal")
    # rows["user_annotation_order"] = rows["user_annotation_order"] - 1

    return rows


def assign_annotations(
    data: pd.DataFrame,
    annotations: pd.DataFrame,
    first_annotation_rule: Callable,
    next_annotations_rule: Callable,
    max_annotations_per_user: Optional[int] = None,
    column_name: str = ''
) -> pd.DataFrame:
    if max_annotations_per_user is None:
        max_annotations_per_user = len(annotations())
    assert max_annotations_per_user > 0, f"Max annotations per user must be grater than 0 but got {max_annotations_per_user}"

    # create ID column
    annotations['user_annotation_order'] = -1
    annotations_column = annotations.columns

    # merge text and annotations
    merged_annotations = annotations.merge(data['text_id'], on='text_id')

    # create first annotations
    merged_annotations = first_annotation_rule(column_name, merged_annotations)

    # create second annotations
    merged_annotations = next_annotations_rule(column_name, merged_annotations, max_annotations_per_user)

    annotations_with_id = merged_annotations[annotations_column]
    return annotations_with_id

# the metrics

def num_of_annotations(column_name: string, data: pd.DataFrame):
    text_df = data.groupby(['text_id'])['annotator_id'].count().reset_index(name="annotations_count")
    data = data.merge(text_df, on='text_id')
    
    return data

def _entropy(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def var_ratio(column_name: string, data: pd.DataFrame):
# def variation_ratio(self, pred_matrix):
    #  """Computes and returns the variation ratios of the predictions in the given prediction matrix"""
    annotations_count_df = data.groupby(['text_id'])['annotator_id'].count().reset_index(name='annotations_count')
    majority_votes_df = data.groupby(["text_id"]).agg(
            majority_votes_df=(column_name, pd.Series.mode))
    var_ratio_df = pd.merge(annotations_count_df, majority_votes_df, on='text_id')
    var_ratio_df["var_ratio"] = 1 - (var_ratio_df.iloc[:,2] / var_ratio_df.iloc[:,1])
    data = data.merge(var_ratio_df[["text_id", "var_ratio"]], on="text_id")
    data['var_ratio'] = [np.array(x).mean() for x in data['var_ratio'].values]
    data['measure_value'] = data['var_ratio']
    data = data.groupby("annotator_id").apply(rank)
    # preds = [preds.argmax(1) for preds in data]
    # mode = Stats.mode(preds, axis=1)
    # var_value = 1 - (mode[1].squeeze() / column_name.T)
    #data = data.groupby(["annotator_id"]).apply(lambda x: rank(x, column=))
    return data

# TODO 
# correct this

def get_entropy(column_name: str, annotations: pd.DataFrame, annotation_columns: List[str]=[], mean=False):

    return _get_text_controversy(column_name, annotations, annotation_columns, _entropy, mean)


def _get_text_controversy(column_name: string, annotations: pd.DataFrame, annotation_columns: List[str], method: Callable, mean: bool):
    temp_df = annotations.copy()
    texts_controversy_df = temp_df.loc[:, ['text_id']].drop_duplicates().reset_index(drop=True)
    # if isinstance(annotation_columns, str):
    #     annotation_columns = [annotation_columns]
    annotation_columns = [column_name]
    controversy_columns = [col + '_controversy' for col in annotation_columns]
    for annotation_col, controversy_col in zip(annotation_columns, controversy_columns):
        text_controversy_dict = annotations.groupby('text_id')[annotation_col].apply(method).to_dict()
        texts_controversy_df[controversy_col] = texts_controversy_df.text_id.apply(text_controversy_dict.get)

    if mean:
        texts_controversy_df['mean_controversy'] = texts_controversy_df.loc[:, controversy_columns].mean(axis=1)
        texts_controversy_df = texts_controversy_df.loc[:, ['annotator_id', 'text_id', 'mean_controversy']]

    # print(f'text_df id count: {len(texts_controversy_df["text_id"])}', 
    #       f'annotation df id count: {len(annotations["text_id"])}',
    #       f'text df annotation df diff ids: {len(set(annotations["text_id"]).intersection(set(texts_controversy_df["text_id"])))}',
    #       sep='\n')
    # texts_controversy_df = texts_controversy_df.groupby(["annotator_id"]).apply(rank)
    annotations = annotations.merge(texts_controversy_df, on="text_id")
    annotations['measure_value'] = annotations[f'{column_name}_controversy']
    annotations = annotations.groupby("annotator_id").apply(rank)

    return annotations

# TODO 
# correct this

def get_weighted_text_controversy(column_name: string, annotations: pd.DataFrame, method: Callable = _entropy, mean: bool = False):
    temp_df = annotations.copy()
    texts_controversy_df = temp_df.loc[:, ['text_id']].drop_duplicates().reset_index(drop=True)
    # if isinstance(annotation_columns, str):
    #     annotation_columns = [annotation_columns]
    annotation_columns = [column_name]
    controversy_columns = [col + '_controversy' for col in annotation_columns]
    for annotation_col, controversy_col in zip(annotation_columns, controversy_columns):
        text_controversy_dict = annotations.groupby('text_id')[annotation_col].apply(method).to_dict()
        texts_controversy_df[controversy_col] = texts_controversy_df.text_id.apply(text_controversy_dict.get)

        
        texts_controversy_df[f"{annotation_col}_annotations_count"] = num_of_annotations(column_name, temp_df)['annotations_count']
        # print(list(texts_controversy_df.columns))
        texts_controversy_df[f"{annotation_col}_annotations_count_norm"] = MinMaxScaler().fit_transform(np.array(texts_controversy_df[f"{annotation_col}_annotations_count"]).reshape(-1,1))
        texts_controversy_df[f"{annotation_col}_weighted_controversy"] = texts_controversy_df[f"{annotation_col}_annotations_count_norm"] * texts_controversy_df[controversy_col]
    #Liczba anotacji per text, znormalizować te wartości i mnożyć każde controversy przez znormalizowaną liczbę anotacji
    
    if mean:
        weighted_controversy_columns = [col + '_weighted_controversy' for col in annotation_columns]
        texts_controversy_df['weighted_controversy'] = texts_controversy_df.loc[:, weighted_controversy_columns].mean(axis=1)
        texts_controversy_df = texts_controversy_df.loc[:, ['annotator_id', 'text_id', 'weighted_controversy']]
    
    annotations = annotations.merge(texts_controversy_df, on="text_id")
    annotations['measure_value'] = annotations[f'{column_name}_weighted_controversy']
    annotations = annotations.groupby("annotator_id").apply(rank)
    return annotations

def get_conformity(column_name: string, annotations: pd.DataFrame = None) -> pd.DataFrame:
        """Computes conformity for each annotator. Works only for binary classification problems."""
        # if annotations is None:
        #     annotations = self.annotations

        df = annotations.copy()
        # column = self.annotation_column

        mean_score = df.groupby("text_id").agg(score_mean=(column_name, "mean"))
        df = df.merge(mean_score.reset_index())

        df["text_major_vote"] = (df["score_mean"] > 0.5).astype(int)

        df["is_major_vote"] = df["text_major_vote"] == df[column_name]
        df["is_major_vote"] = df["is_major_vote"].astype(int)

        positive_df = df[df.text_major_vote == 1]
        negative_df = df[df.text_major_vote == 0]

        conformity_df = df.groupby("annotator_id").agg(
            conformity=("is_major_vote", "mean")
        )
        conformity_df["pos_conformity"] = positive_df.groupby("annotator_id").agg(
            pos_conformity=("is_major_vote", "mean")
        )
        conformity_df["neg_conformity"] = negative_df.groupby("annotator_id").agg(
            neg_conformity=("is_major_vote", "mean")
        )

        # conformity_df = conformity_df.groupby(["annotator_id"]).apply(rank(conformity_df[conformity_df['text_id']], 'annotator_id'))
        annotations = annotations.merge(conformity_df, on="annotator_id")
        text_conformity_df = annotations.groupby('text_id').agg(text_mean_conformity=('conformity', "mean"))
        annotations = annotations.merge(text_conformity_df, on="text_id")
        annotations['measure_value'] = annotations['text_mean_conformity']
        annotations = annotations.groupby("annotator_id").apply(rank)
        return annotations


def get_weighted_conformity(column_name: string, annotations: pd.DataFrame = None) -> pd.DataFrame:
        """Computes conformity for each annotator. Works only for binary classification problems."""
        # if annotations is None:
        #     annotations = self.annotations
        df = annotations.copy()
        # column = self.annotation_column
        mean_score = df.groupby("text_id").agg(score_mean=(column_name, "mean"))
        df = df.merge(mean_score.reset_index())
        df["text_major_vote"] = (df["score_mean"] > 0.5).astype(int)
        df['annotation_group_ratio'] = 0.0
        df.loc[df[column_name] == 1, 'annotation_group_ratio'] = df["score_mean"] 
        df.loc[df[column_name] == 0, 'annotation_group_ratio'] = 1 - df["score_mean"] 
        positive_df = df[df.text_major_vote == 1]
        negative_df = df[df.text_major_vote == 0]
        conformity_df = df.groupby("annotator_id").agg(
            weighted_conformity=("annotation_group_ratio", "mean")
        )
        conformity_df["pos_conformity"] = positive_df.groupby("annotator_id").agg(
            pos_weighted_conformity=("annotation_group_ratio", "mean")
        )
        conformity_df["neg_conformity"] = negative_df.groupby("annotator_id").agg(
            neg_weighted_conformity=("annotation_group_ratio", "mean")
        )
        # conformity_df = conformity_df.groupby(["annotator_id"]).apply(lambda x: rank(x, column='conformity'))
        # raise Exception(f'{pd.unique(conformity_df["user_annotation_order"])}')
        # annotations.merge(conformity_df, on="annotator_id")
        annotations = annotations.merge(conformity_df, on="annotator_id")
        text_conformity_df = annotations.groupby('text_id').agg(text_mean_weighted_conformity=('weighted_conformity', "mean"))
        annotations = annotations.merge(text_conformity_df, on="text_id")
        annotations['measure_value'] = annotations['text_mean_weighted_conformity']
        annotations = annotations.groupby("annotator_id").apply(rank)



        # conformity_df['measure_value'] = conformity_df['weighted_conformity']
        # conformity_df = conformity_df.groupby("annotator_id").apply(rank)
        return annotations



def get_annotation_count_weighted_weighted_conformity(column_name: string, annotations: pd.DataFrame = None) -> pd.DataFrame:
        """Computes conformity for each annotator. Works only for binary classification problems."""
        # if annotations is None:
        #     annotations = self.annotations
        df = annotations.copy()
        # column = self.annotation_column
        mean_score = df.groupby("text_id").agg(score_mean=(column_name, "mean"))
        df = df.merge(mean_score.reset_index())
        df["text_major_vote"] = (df["score_mean"] > 0.5).astype(int)
        df['annotation_group_ratio'] = 0.0
        df.loc[df[column_name] == 1, 'annotation_group_ratio'] = df["score_mean"] 
        df.loc[df[column_name] == 0, 'annotation_group_ratio'] = 1 - df["score_mean"] 
        positive_df = df[df.text_major_vote == 1]
        negative_df = df[df.text_major_vote == 0]
        conformity_df = df.groupby("annotator_id").agg(
            weighted_conformity=("annotation_group_ratio", "mean")
        )
        conformity_df["pos_conformity"] = positive_df.groupby("annotator_id").agg(
            pos_weighted_conformity=("annotation_group_ratio", "mean")
        )
        conformity_df["neg_conformity"] = negative_df.groupby("annotator_id").agg(
            neg_weighted_conformity=("annotation_group_ratio", "mean")
        )
        
        # conformity_df = conformity_df.groupby(["annotator_id"]).apply(lambda x: rank(x, column='conformity'))
        # raise Exception(f'{pd.unique(conformity_df["user_annotation_order"])}')
        # annotations.merge(conformity_df, on="annotator_id")
        annotations = annotations.merge(conformity_df, on="annotator_id")
        text_conformity_df = annotations.groupby('text_id').agg(text_mean_weighted_conformity=('text_weighted_conformity', "mean"))

        annotations_df = num_of_annotations(column_name, annotations.copy())
        text_conformity_df.merge(annotations_df, on='text_id')
        # print(list(texts_controversy_df.columns))
        text_conformity_df[f"{column_name}_annotations_count_norm"] = MinMaxScaler().fit_transform(np.array(text_conformity_df[f"{column_name}_annotations_count"]).reshape(-1,1))
        text_conformity_df[f"{column_name}_weighted_controversy"] = text_conformity_df[f"{column_name}_annotations_count_norm"] * text_conformity_df[column_name]

        annotations = annotations.merge(text_conformity_df, on="text_id")
        annotations['measure_value'] = annotations['text_mean_weighted_conformity']
        annotations = annotations.groupby("annotator_id").apply(rank)
        # conformity_df['measure_value'] = conformity_df['weighted_conformity']
        # conformity_df = conformity_df.groupby("annotator_id").apply(rank)
        return annotations


def get_min_conformity(column_name: string, annotations: pd.DataFrame = None) -> pd.DataFrame:
        """Computes conformity for each annotator. Works only for binary classification problems."""
        # if annotations is None:
        #     annotations = self.annotations

        df = annotations.copy()
        # column = self.annotation_column

        mean_score = df.groupby("text_id").agg(score_mean=(column_name, "mean"))
        df = df.merge(mean_score.reset_index())

        df["text_major_vote"] = (df["score_mean"].max()).astype(int)

        df["is_major_vote"] = df["text_major_vote"] == df[column_name]
        df["is_major_vote"] = df["is_major_vote"].astype(int)

        positive_df = df[df.text_major_vote == 1]
        negative_df = df[df.text_major_vote == 0]

        conformity_df = df.groupby("annotator_id").agg(
            conformity=("is_major_vote", "mean")
        )
        conformity_df["pos_conformity"] = positive_df.groupby("annotator_id").agg(
            pos_conformity=("is_major_vote", "mean")
        )
        conformity_df["neg_conformity"] = negative_df.groupby("annotator_id").agg(
            neg_conformity=("is_major_vote", "mean")
        )
        min_user_conformity = conformity_df.groupby("annotator_id").agg(
            min_text_conformity=("conformity", "min")
        )
        conformity_df = conformity_df.merge(min_user_conformity, on="annotator_id")
        annotations = annotations.merge(conformity_df, on="annotator_id")
        text_conformity_df = annotations.groupby('text_id').agg(text_mean_min_conformity=('min_text_conformity', "mean"))
        annotations = annotations.merge(text_conformity_df, on="text_id")
        # conformity_df = conformity_df.groupby(["annotator_id"]).apply(rank)
        # annotations.join(conformity_df, on="annotator_id")
        annotations['measure_value'] = annotations['text_mean_min_conformity']
        annotations = annotations.groupby("annotator_id").apply(rank)
        return annotations


def get_mean_conformity(column_name: string, annotations: pd.DataFrame = None) -> pd.DataFrame:
        """Computes conformity for each annotator. Works only for binary classification problems."""
        # if annotations is None:
        #     annotations = self.annotations

        df = annotations.copy()
        # column = self.annotation_column

        mean_score = df.groupby("text_id").agg(score_mean=(column_name, "mean"))
        df = df.merge(mean_score.reset_index())

        df["text_major_vote"] = (df["score_mean"] > 0.5).astype(int)

        df["is_major_vote"] = df["text_major_vote"] == df[column_name]
        df["is_major_vote"] = df["is_major_vote"].astype(int)

        positive_df = df[df.text_major_vote == 1]
        negative_df = df[df.text_major_vote == 0]

        # count_score = df.groupby("text_id").agg()

        conformity_df = df.groupby("annotator_id").agg(
            conformity=("is_major_vote", "mean")
        )
        conformity_df["pos_conformity"] = positive_df.groupby("annotator_id").agg(
            pos_conformity=("is_major_vote", "mean")
        )
        conformity_df["neg_conformity"] = negative_df.groupby("annotator_id").agg(
            neg_conformity=("is_major_vote", "mean")
        )
        mean_user_conformity = conformity_df.groupby("annotator_id").agg(
            mean_text_conformity=("conformity", "mean")
        )
        conformity_df = conformity_df.merge(mean_user_conformity, on="annotator_id")
        annotations = annotations.merge(conformity_df, on="annotator_id")
        text_conformity_df = annotations.groupby('text_id').agg(text_mean_mean_conformity=('mean_text_conformity', "mean"))
        annotations = annotations.merge(text_conformity_df, on="text_id")
        # conformity_df = conformity_df.groupby(["annotator_id"]).apply(rank)
        # annotations.join(conformity_df, on="annotator_id")
        annotations['measure_value'] = annotations['text_mean_mean_conformity']
        annotations = annotations.groupby("annotator_id").apply(rank)
        return annotations


def neighbour_annotators_count(column_name: str, annotations: pd.DataFrame = None) -> pd.DataFrame:

    #Zliczamy teksty po użytkowniku
    #bierzemy teksty, które jeszcze nie zostały przez niego ocenione
    #wartośc miary = lista anotatorów, z któymi użytkownik niezaanotował, robimy unikatowość, mamy czarną listę
    #lecimy po każdej anotacji, patrzymy kto anotował ten takst, którego dotyczy poszczególna anotacja(user_annotation_order) dodajemy ich do czarnej liscie
    #Lista autorów, z któymi pisał coś(to ta czarna lista)
    #Liczba niepowtarzających się użytkowników dla każdej anotacji
    #z poprzedniego zliczenia zbieramy użytkowników

    return 

# identity measure
def identity(x, *args, **kwargs):
    return x