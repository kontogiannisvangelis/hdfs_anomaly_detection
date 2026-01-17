import numpy as np
from src.dataset.dataset_loader import load_npz
def main():
    df = load_npz("../../data/NPZ/HDFS.npz", "../../data/HDFS_templates.csv", True)

    lengths = df['text'].str.len().tolist()

    print(df.head())

    min_length = np.min(lengths)
    max_length = np.max(lengths)
    mean_length = np.mean(lengths)
    median_length = np.median(lengths)
    percentile_95 = np.percentile(lengths, 95)

    print(f"Total Samples (Train Split): {len(df)}")
    print(f"Min Sequence Length: {min_length}")
    print(f"Max Sequence Length: {max_length}")
    print(f"Mean Sequence Length: {mean_length:.2f}")
    print(f"Median Sequence Length: {median_length}")
    print(f"95th Percentile Length: {percentile_95}")

if __name__ == "__main__":
    main()