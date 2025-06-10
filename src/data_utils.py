import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data from the given file path."""
    df = pd.read_csv(file_path)
    print(f"[INFO] Data loaded from {file_path}")
    return df

def show_basic_info(df: pd.DataFrame) -> None:
    """Display basic information about the DataFrame."""
    print("[INFO] Shape:", df.shape)
    print("[INFO] Data Types:\n", df.dtypes)
    print("[INFO] Missing Values:\n", df.isnull().sum())

def main():
    # Change this path to your actual file location
    data_file = 'C:/Users/M338548/Documents/git/gait-analysis-one-day/data/synthetic_gait_analysis_dataset.csv'
    gait_df = load_data(data_file)
    show_basic_info(gait_df)

if __name__ == "__main__":
    main()
