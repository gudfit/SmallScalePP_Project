import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_performance_file(filepath):
    """Parses a performance output file and extracts data."""
    filename = os.path.basename(filepath)
    method_name = filename.replace("output_", "").replace(".txt", "")

    data = []
    current_n = None
    current_k = None

    with open(filepath, "r") as f:
        for line in f:
            n_match = re.search(r"Testing: n = (\d+), k = (\d+)", line)
            if n_match:
                current_n = int(n_match.group(1))
                current_k = int(n_match.group(2))

            perf_match = re.search(r"Performance:\s+([\d.]+)\s+GFLOPS", line)
            if perf_match and current_n is not None and current_k is not None:
                performance = float(perf_match.group(1))
                data.append(
                    {
                        "Method": method_name,
                        "n": current_n,
                        "k": current_k,
                        "Performance (GFLOPS)": performance,
                    }
                )
                current_n = None

    return data


def create_performance_plot(df, n_value, output_filename):
    """Creates and saves a line plot of performance data for a specific n."""

    n_df = df[df["n"] == n_value]

    if n_df.empty:
        print(f"No data to plot for n = {n_value}")
        return

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=n_df,
        x="k",
        y="Performance (GFLOPS)",
        hue="Method",
        style="Method",
        marker="o",
    )

    plt.title(f"Matrix Multiplication Performance (n={n_value})")
    plt.xlabel("k (Dimension K)")
    plt.ylabel("Performance (GFLOPS)")
    plt.grid(True)
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Performance plot saved to: {output_filename}")


if __name__ == "__main__":
    root_dir = "."
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    data_list = []

    for subdir in ["cudasrc", "openmpsrc", "output"]:
        dir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(dir_path):
            continue

        for filename in os.listdir(dir_path):
            if filename.startswith("output_") and filename.endswith(".txt"):
                filepath = os.path.join(dir_path, filename)
                parsed_data = parse_performance_file(filepath)
                data_list.extend(parsed_data)

    if not data_list:
        print("No output_*.txt files found in cudasrc or openmpsrc directories.")
    else:
        df = pd.DataFrame(data_list)
        n_values = sorted(df["n"].unique())

        for n_val in n_values:
            output_filename = os.path.join(
                results_dir, f"performance_plot_n_{n_val}.png"
            )
            create_performance_plot(df, n_val, output_filename)

        print(
            "Data processing and plotting complete. Plots saved in 'results' directory."
        )
