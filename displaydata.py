import pandas as pd
import matplotlib.pyplot as plt

def plot_pose_averages(csv_file='pose_metrics.csv'):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    if df.empty:
        print("No data to plot.")
        return

    # Find all unique segment/stat columns
    stat_types = ['angle_deg', 'velocity_m_s', 'torque_Nm']
    segments = set()
    for col in df.columns:
        for stat in stat_types:
            if col.endswith(stat):
                segments.add(col.replace(f"_{stat}", ""))
    segments = sorted(list(segments))

    # Plot each stat type for all segments
    for stat in stat_types:
        plt.figure(figsize=(14, 6))
        for seg in segments:
            col = f"{seg}_{stat}"
            if col in df:
                plt.plot(df[col], label=seg)
        plt.title(f"Running Average: {stat}")
        plt.xlabel("Logged Sample")
        plt.ylabel(stat)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_pose_averages('pose_metrics.csv')
