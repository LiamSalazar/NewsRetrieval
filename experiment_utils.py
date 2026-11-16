from pathlib import Path
import pandas as pd


def update_experiment_log(
    new_rows,
    tasks_to_replace
):

    """
    Replace only the specified task records.

    This preserves experiments belonging to other tasks
    and prevents duplicated records after rerunning scripts.
    """

    path = Path("experiments.tsv")

    new_df = pd.DataFrame(new_rows)

    if path.exists():

        old_df = pd.read_csv(
            path,
            sep="\t"
        )

        if "task" in old_df.columns:

            old_df = old_df.loc[
                ~old_df["task"].isin(tasks_to_replace)
            ].copy()

        combined = pd.concat(
            [old_df, new_df],
            ignore_index=True,
            sort=False
        )

    else:

        combined = new_df

    combined.to_csv(
        path,
        sep="\t",
        index=False
    )


def safe_text(value, max_chars=500):

    if pd.isna(value):
        return ""

    return (
        str(value)
        .replace("\t", " ")
        .replace("\n", " ")
    )[:max_chars]
