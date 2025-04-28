import os
import sys

# Add components directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "components"))

from stackoverflow_preprocessor import DataProcessor

if __name__ == "__main__":
    # Dynamically locate the data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Parent directory
    data_dir = os.path.join(base_dir, "data")

    processor = DataProcessor(data_dir=data_dir)
    processor.load_data()
    processor.calculate_influence_score()
    processor.clean_data()
    processor.save_preprocessed_data()
    processor.merge_data()


from reddit_preprocessor import RedditPreprocessor

if __name__ == "__main__":
    # Locate parent directory and data directory dynamically
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    processor = RedditPreprocessor(
        data_dir=data_dir,
        posts_file="askreddit_posts.csv",
        comments_file="askreddit_comments.csv",
        n_keep=80_000,
        random_sample=False
    )
    processor.run_all()