import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib


def plot_state_steps(
    dataframe: pd.DataFrame,
    model_config,
) -> None:
    """Plots the dynamics of state components over time with rewards annotated.

    This function visualizes the evolution of state components over a series of steps.
    Each state component is plotted separately, and rewards are annotated at specific
    intervals for better interpretation.

    Args:
        dataframe (pd.DataFrame): A DataFrame containing the transitions data,
            including columns for "state" and "reward".
        model_config: Configuration object containing the input state columns.
    """
    idx2label = dict(enumerate(model_config.input.state_columns))

    data = dataframe.to_dict(orient="records")

    states = np.array([d["state"] for d in data])
    rewards = np.array([d["reward"] for d in data])

    steps = np.arange(len(states))

    fig, axes = plt.subplots(nrows=states.shape[1], ncols=1, figsize=(13.33, 7.5 * states.shape[1]), dpi=300)

    for component_idx in range(states.shape[1]):
        label = idx2label[component_idx]

        state_component = states[:, component_idx]

        ax = axes[component_idx]

        ax.plot(steps, state_component, marker='o', color='blue', label=f"Dynamics of {label}")

        for i in range(0, len(steps), 4):
            ax.text(steps[i], state_component[i], f"R={rewards[i]:.2f}", fontsize=9, color='red', ha='right')

        ax.legend(fontsize=12)
        ax.set_xlabel("Step", fontsize=14)
        ax.set_ylabel(f"{label}", fontsize=14)
        ax.set_xticks(steps[::4])
        ax.set_yticks(np.arange(np.min(state_component), np.max(state_component)))
        ax.set_xticklabels(ax.get_xticks(), rotation=90)
        ax.grid(True)

    plt.show()


def plot_action_steps(
    dataframe: pd.DataFrame,
    model_config,
) -> None:
    """Plots the dynamics of action components over time with rewards annotated.

    This function visualizes the cumulative actions over a series of steps.
    Each action component is plotted separately, and rewards are annotated at specific
    intervals for better interpretation.

    Args:
        dataframe (pd.DataFrame): A DataFrame containing the transitions data,
            including columns for "action" and "reward".
        model_config: Configuration object containing the input action columns.
    """
    idx2label = dict(enumerate(model_config.input.action_columns))

    data = dataframe.to_dict(orient="records")

    actions = np.array([d["action"] for d in data])
    actions = np.cumsum(actions)

    if len(actions.shape) == 1:
        actions = actions.reshape(-1, 1)

    rewards = np.array([d["reward"] for d in data])

    steps = np.arange(len(actions))

    fig, axes = plt.subplots(nrows=actions.shape[1], ncols=1, figsize=(13.33, 7.5 * actions.shape[1]), dpi=300)

    if isinstance(axes, matplotlib.axes._axes.Axes):
        axes = np.array([axes])

    for component_idx in range(actions.shape[1]):
        label = idx2label[component_idx]

        action_component = actions[:, component_idx]

        ax = axes[component_idx]

        ax.plot(steps, action_component, marker='o', color='blue', label=f"Dynamics of {label}")

        for i in range(0, len(steps), 4):
            ax.text(steps[i], action_component[i], f"R={rewards[i]:.2f}", fontsize=9, color='red', ha='right')

        ax.legend(fontsize=12)
        ax.set_xlabel("Step", fontsize=14)
        ax.set_ylabel(f"{label}", fontsize=14)
        ax.set_xticks(steps[::4])
        ax.set_yticks(np.arange(np.min(action_component), np.max(action_component)))
        ax.set_xticklabels(ax.get_xticks(), rotation=90)
        ax.grid(True)

    plt.show()


def plot_rewards(dataframe: pd.DataFrame) -> None:
    """Plots the dynamics of rewards over time.

    This function visualizes the evolution of rewards over a series of steps.
    Rewards are plotted with markers and annotated at specific intervals.

    Args:
        dataframe (pd.DataFrame): A DataFrame containing the transitions data,
            including a column for "reward".
    """
    data = dataframe.to_dict(orient="records")

    rewards = np.array([d["reward"] for d in data])

    steps = np.arange(len(rewards))

    plt.figure(figsize=(13.33, 7.5), dpi=300)
    plt.plot(steps, rewards, marker='o', color='blue', label=f"Dynamics of Reward")

    plt.legend(fontsize=12)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel(f"Reward", fontsize=14)
    plt.xticks(steps[::4], rotation=90)
    plt.yticks(np.arange(np.min(rewards), np.max(rewards)))
    plt.grid(True)

    plt.show()