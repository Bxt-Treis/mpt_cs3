from ax.service.ax_client import AxClient, ObjectiveProperties
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import polars as pl


def main():
    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=[
            {"name": "forming_speed", "type": "range", "bounds": [50.0, 300.0]},
            {
                "name": "blank_holding_force_start",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "blank_holding_force_middle",
                "type": "range",
                "bounds": [1.0, 2.0],
            },
            {
                "name": "blank_holding_force_end",
                "type": "range",
                "bounds": [2.0, 5.0],
            },
        ],
        objectives={
            "thinning": ObjectiveProperties(minimize=True),
            "wrinkling": ObjectiveProperties(minimize=True),
        },
    )
    # define acquisition function to be upper confidence bound

    folder = Path("results")
    base = "result"

    for i in range(3):
        full_path = folder / f"{base}_{i}.txt"
        if full_path.is_file():
            df = pl.read_csv(full_path)
            print("found results")
            for row in df.iter_rows():
                _, trial_index = ax_client.attach_trial(
                    parameters={
                        "forming_speed": row[0],
                        "blank_holding_force_start": row[1],
                        "blank_holding_force_middle": row[2],
                        "blank_holding_force_end": row[3],
                    }
                )
                ax_client.complete_trial(
                    trial_index=trial_index,
                    raw_data={"thinning": row[4], "wrinkling": row[5]},
                )
            continue
        else:
            params, trial_index = ax_client.get_next_trial()
            print(params, trial_index)

    # Plot Pareto frontier
    frontier = ax_client.get_pareto_optimal_parameters()

    thinning_vals = []
    wrinkling_vals = []
    for trial_index, (params, (means, covariances)) in frontier.items():
        thinning_vals.append(means["thinning"])
        wrinkling_vals.append(means["wrinkling"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=thinning_vals,
            y=wrinkling_vals,
            mode="markers+lines",
            name="Pareto Frontier",
        )
    )
    fig.update_layout(
        title="Pareto Frontier: Thinning vs Wrinkling",
        xaxis_title="Thinning",
        yaxis_title="Wrinkling",
    )
    fig.show()


if __name__ == "__main__":
    main()
