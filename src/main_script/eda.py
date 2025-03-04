import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.eda import EDA

if __name__ == "__main__":
    file_path = "../data/preprocessed_data.csv"
    eda = EDA(file_path)
    eda.load_data()
    eda.preprocess_data(["user_id", "reputation", "display_name"])
    eda.plot_heatmap()