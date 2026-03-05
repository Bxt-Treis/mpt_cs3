from ax.service.ax_client import AxClient, ObjectiveProperties
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import polars as pl


def main():
    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=[
            {"name": "forming_speed", "type": "range", "bounds": [100.0, 300.0]},
            {
                "name": "blank_holding_force_start",
                "type": "range",
                "bounds": [100.0, 225.0],
            },
            {
                "name": "blank_holding_force_middle",
                "type": "range",
                "bounds": [225.0, 375.0],
            },
            {
                "name": "blank_holding_force_end",
                "type": "range",
                "bounds": [375.0, 500.0],
            },
        ],
        objectives={
            "thinning": ObjectiveProperties(minimize=True, threshold=0.4),
            "wrinkling": ObjectiveProperties(minimize=True, threshold=70.0),
        },
    )
    # define acquisition function to be upper confidence bound

    folder = Path("results")
    base = "result"

    for i in range(3):
        full_path = folder / f"{base}_{i}.csv"
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
                    raw_data={
                        "thinning": abs(row[4]),
                        "wrinkling": row[5],
                    },
                )
            continue
        else:
            generator_runs, _ = ax_client.get_next_trials(max_trials=9)
            for trial_index, params in generator_runs.items():
                print(f"Trial {trial_index}: {params}")
            return

    # Plot Pareto frontier
    frontier = ax_client.get_pareto_optimal_parameters()

    thinning_vals = []
    wrinkling_vals = []
    for trial_index, (params, (means, covariances)) in frontier.items():
        thinning_vals.append(means["thinning"])
        wrinkling_vals.append(means["wrinkling"])

    # 3D Pareto surface
    fig_3d = go.Figure()
    fig_3d.add_trace(
        go.Scatter3d(
            x=thinning_vals,
            y=wrinkling_vals,
            mode="markers",
            marker=dict(size=5),
            name="Pareto Frontier",
        )
    )
    fig_3d.update_layout(
        title="Pareto Frontier: Thinning vs Wrinkling vs CO2",
        scene=dict(
            xaxis_title="Thinning",
            yaxis_title="Wrinkling",
        ),
    )
    fig_3d.write_image("pareto_frontier_3d.svg")

    # Pairwise 2D plots
    pairs = [
        ("Thinning", "Wrinkling", thinning_vals, wrinkling_vals),
    ]
    for x_label, y_label, x_data, y_data in pairs:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=x_data, y=y_data, mode="markers", name="Pareto Frontier")
        )
        fig.update_layout(
            title=f"Pareto Frontier: {x_label} vs {y_label}",
            xaxis_title=x_label,
            yaxis_title=y_label,
        )
        fig.write_image(f"pareto_{x_label.lower()}_vs_{y_label.lower()}.svg")


if __name__ == "__main__":
    main()
