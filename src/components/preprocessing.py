# preprocessing.py
import os
import math
import pandas as pd
import matplotlib.pyplot as plt


class DataProcessor:
    """A class to preprocess and merge Stack Overflow user data."""

    def __init__(self, users_file, questions_file, answers_file, output_dir="data"):
        """Initialize file paths and output directory."""
        self.users_file = users_file
        self.questions_file = questions_file
        self.answers_file = answers_file
        self.output_dir = output_dir  # Save processed files in 'data' directory
        self.df_users = None
        self.df_questions = None
        self.df_answers = None

    def load_data(self):
        """Load CSV files into Pandas DataFrames."""
        self.df_users = pd.read_csv(self.users_file)
        self.df_questions = pd.read_csv(self.questions_file)
        self.df_answers = pd.read_csv(self.answers_file)

    @staticmethod
    def remove_columns(df: pd.DataFrame, columns_to_remove: list):
        """Remove specified columns from a DataFrame, ignoring any that don't exist."""
        df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True, errors="ignore")

    def calculate_influence_score(self):
        """Compute the influence score for users and add an 'influential' binary column."""
        self.df_users['influence_score'] = self.df_users.apply(
            lambda row: math.log1p(row["reputation"])
                        + 3 * math.log1p(row["gold_badges"])
                        + 2 * math.log1p(row["silver_badges"])
                        + math.log1p(row["bronze_badges"]),
            axis=1
        )

        # Compute the threshold (90th percentile)
        threshold = self.df_users["influence_score"].quantile(0.90)
        print(f"Influence Score Threshold: {threshold}")

        # Binarize 'influential' column
        self.df_users["influential"] = (self.df_users["influence_score"] > threshold).astype(int)

        # Plot influence score distribution
        plt.figure(figsize=(8, 6))
        self.df_users["influence_score"].hist(bins=30, edgecolor="black")
        plt.title("Distribution of Influence Score (Histogram)")
        plt.xlabel("Influence Score")
        plt.ylabel("Frequency")
        plt.show()

    def clean_data(self):
        """Remove unnecessary columns from the datasets."""
        # Remove specified columns from users, questions, and answers datasets
        self.remove_columns(self.df_users, ["reputation", "gold_badges", "silver_badges", "bronze_badges", "influence_score"])
        self.remove_columns(self.df_questions, ["reputation", "gold_badges", "silver_badges", "bronze_badges"])
        self.remove_columns(self.df_answers, ["reputation", "gold_badges", "silver_badges", "bronze_badges"])

    def save_preprocessed_data(self):
        """Save cleaned data to CSV files in the existing data directory."""
        self.df_users.to_csv(f"{self.output_dir}/preprocessed_users.csv", index=False)
        self.df_questions.to_csv(f"{self.output_dir}/preprocessed_questions.csv", index=False)
        self.df_answers.to_csv(f"{self.output_dir}/preprocessed_answers.csv", index=False)
        print("Preprocessed data saved.")

    def merge_data(self):
        """Merge the cleaned datasets by aggregating questions and answers."""
        # Ensure 'is_accepted' column is treated as boolean
        self.df_answers['is_accepted'] = self.df_answers['is_accepted'].astype(bool)

        # Aggregate questions dataset
        questions_agg = self.df_questions.groupby('user_id').agg(
            total_questions=('question_id', 'count'),
            avg_question_score=('score', 'mean'),
        ).reset_index()

        # Aggregate answers dataset
        answers_agg = self.df_answers.groupby('user_id').agg(
            avg_answer_score=('score', 'mean'),
            accepted_answers=('is_accepted', 'sum')
        ).reset_index()

        # Merge aggregated data with the users dataset
        merged_df = self.df_users.merge(questions_agg, on='user_id', how='left')
        merged_df = merged_df.merge(answers_agg, on='user_id', how='left')

        # Replace NaN values with 0 for users without questions or answers
        cols_to_fill = ['total_questions', 'avg_question_score', 'avg_answer_score', 'accepted_answers']
        merged_df[cols_to_fill] = merged_df[cols_to_fill].fillna(0)

        # Save the merged dataset in the data directory
        merged_df.to_csv(f"{self.output_dir}/preprocessed_data.csv", index=False)
        print("Merged dataset saved to data/preprocessed_data.csv")
