import pandas as pd


def stratify_by_users(sorted_not_annotated: pd.DataFrame, amount_per_user: int):
    filtered_annotations = sorted_not_annotated.groupby("annotator_id").apply(
        lambda rows: rows[:amount_per_user]
    )

    filtered_annotations = filtered_annotations.reset_index(drop=True)
    filtered_annotations = filtered_annotations.loc[:, ["text_id", "annotator_id"]]

    if not "original_index" in sorted_not_annotated.columns:
        sorted_not_annotated = sorted_not_annotated.reset_index()

    sorted_not_annotated = sorted_not_annotated.merge(filtered_annotations)
    return sorted_not_annotated.set_index("original_index")


def stratify_by_users_decorator(amount_per_user: int):
    def _stratify_by_users(sort_annotations_func):
        def wrapper(*args, **kwargs):
            sorted_annotations = sort_annotations_func(*args, **kwargs)

            filtered_annotations = sorted_annotations.groupby("annotator_id").apply(
                lambda rows: rows[:amount_per_user]
            )

            filtered_annotations = filtered_annotations.reset_index(drop=True)
            filtered_annotations = filtered_annotations.loc[
                :, ["text_id", "annotator_id"]
            ]

            if not "original_index" in sorted_annotations.columns:
                sorted_annotations = sorted_annotations.reset_index()

            sorted_not_annotated = sorted_annotations.merge(filtered_annotations)

            return sorted_not_annotated.set_index("original_index")

        return wrapper

    return _stratify_by_users
