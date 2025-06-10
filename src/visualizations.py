import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set global Seaborn theme
sns.set(style="whitegrid")


def plot_age_distribution_with_pathology(data: pd.DataFrame):
    """
    Histogram of age distribution overlaid by pathology group.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x="age", hue="pathology", multiple="stack", bins=20, palette="Set2")
    plt.title("Age Distribution by Pathology")
    plt.xlabel("Age (years)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_walking_speed_boxplot(data: pd.DataFrame):
    """
    Box plot of walking speed by pathology group.
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x="pathology", y="walking_speed", palette="Set3")
    plt.title("Walking Speed by Pathology Group")
    plt.xlabel("Pathology Group")
    plt.ylabel("Walking Speed (m/s)")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(data: pd.DataFrame, temporal_features: list[str]):
    """
    Heatmap of correlation between temporal gait parameters.
    """
    temporal_data = data[temporal_features]
    correlation_matrix = temporal_data.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap: Temporal Gait Parameters")
    plt.tight_layout()
    plt.show()


def plot_stride_length_vs_cadence(data: pd.DataFrame):
    """
    Scatter plot of stride length vs cadence, colored by condition.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x="stride_length", y="cadence", hue="condition", palette="Set1", s=60)
    plt.title("Stride Length vs. Cadence by Condition")
    plt.xlabel("Stride Length (m)")
    plt.ylabel("Cadence (steps/min)")
    plt.tight_layout()
    plt.show()


# Example use (uncomment below and ensure your DataFrame is named `df`):
plot_age_distribution_with_pathology(df)
plot_walking_speed_boxplot(df)
plot_correlation_heatmap(df, ['step_time', 'stride_time', 'swing_time', 'stance_time'])
plot_stride_length_vs_cadence(df)
