import pandas as pd
import wandb


def get_wandb_project_rows(project_name):
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(project_name)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )

    return pd.concat(
        [
            pd.DataFrame(runs_df["summary"].tolist()),
            pd.DataFrame(runs_df["config"].tolist()),
        ],
        axis=1,
    )
