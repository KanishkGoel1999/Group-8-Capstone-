import sys
sys.path.append("../components")  # Add components directory to the Python path

from preprocessing import DataProcessor

if __name__ == "__main__":
    processor = DataProcessor(
        users_file="data/stackoverflow_users.csv",
        questions_file="data/stackoverflow_questions.csv",
        answers_file="data/stackoverflow_answers.csv",
        output_dir="data"  # Ensure output files are saved in the data directory
    )
    processor.load_data()
    processor.calculate_influence_score()
    processor.clean_data()
    processor.save_preprocessed_data()
    processor.merge_data()
