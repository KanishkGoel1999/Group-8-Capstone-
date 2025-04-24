import os
import csv
import time
import requests

class StackOverflowDataFetcher:
    BASE_URL = "https://api.stackexchange.com/2.3/"
    HEADERS = {"Accept-Encoding": "gzip"}
    SITE = "stackoverflow"
    PAGE_SIZE = 100

    def __init__(self, api_keys, output_dir, sleep_time=1):
        """
        Initialize the fetcher with API keys and output location.
        :param api_keys: Dict with 'users', 'questions', 'answers' keys and their corresponding API keys.
        :param output_dir: Where to store the resulting CSV files.
        :param sleep_time: Delay between API requests to avoid rate limits.
        """
        self.api_keys = api_keys
        self.output_dir = output_dir
        self.sleep_time = sleep_time
        os.makedirs(output_dir, exist_ok=True)

    def _fetch_paginated_data(self, endpoint, fields, api_key):
        """Fetch paginated data from StackExchange API with backoff handling."""
        data = []
        page = 1

        while True:
            print(f"Fetching page {page} for endpoint '{endpoint}'...")
            params = {
                "site": self.SITE,
                "pagesize": self.PAGE_SIZE,
                "page": page,
                "key": api_key
            }

            response = requests.get(f"{self.BASE_URL}{endpoint}", headers=self.HEADERS, params=params)
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    print("Error:", error_data)
                    if "backoff" in error_data:
                        backoff_time = error_data["backoff"]
                        print(f"Backing off for {backoff_time} seconds...")
                        time.sleep(backoff_time)
                        continue
                except Exception as e:
                    print("Failed to parse error response:", e)
                break

            items = response.json().get("items", [])
            if not items:
                break

            for item in items:
                extracted = {field: item.get(field, "") for field in fields}
                if "owner" in item:
                    extracted.update({
                        "user_id": item["owner"].get("user_id", ""),
                        "display_name": item["owner"].get("display_name", ""),
                        "reputation": item["owner"].get("reputation", 0)
                    })
                data.append(extracted)

            if not response.json().get("has_more", False):
                break

            page += 1
            time.sleep(self.sleep_time)

        return data

    def fetch_users(self):
        users = []
        page = 1
        while True:
            print(f"Fetching user page {page}")
            params = {
                "site": self.SITE,
                "pagesize": self.PAGE_SIZE,
                "page": page,
                "key": self.api_keys["users"]
            }

            response = requests.get(f"{self.BASE_URL}users", headers=self.HEADERS, params=params)
            if response.status_code != 200:
                print("Error fetching users:", response.status_code)
                break

            data = response.json()
            users.extend(data.get("items", []))

            if not data.get("has_more", False):
                break

            page += 1
            time.sleep(self.sleep_time)

        return users

    def save_users_csv(self, filename="stackoverflow_users.csv"):
        users = self.fetch_users()
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["user_id", "display_name", "reputation", "gold_badges", "silver_badges", "bronze_badges"])
            for user in users:
                writer.writerow([
                    user.get("user_id"),
                    user.get("display_name"),
                    user.get("reputation"),
                    user.get("badge_counts", {}).get("gold", 0),
                    user.get("badge_counts", {}).get("silver", 0),
                    user.get("badge_counts", {}).get("bronze", 0)
                ])
        print(f"✅ Users saved to {filepath}")

    def save_questions_csv(self, filename="stackoverflow_questions.csv"):
        fields = ["question_id", "tags", "score", "accepted_answer_id"]
        questions = self._fetch_paginated_data("questions", fields, self.api_keys["questions"])
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=[
                "question_id", "tags", "score", "user_id", "reputation", "display_name", "accepted_answer_id"
            ])
            writer.writeheader()
            writer.writerows(questions)
        print(f"✅ Questions saved to {filepath}")

    def save_answers_csv(self, filename="stackoverflow_answers.csv"):
        fields = ["answer_id", "question_id", "is_accepted", "score"]
        answers = self._fetch_paginated_data("answers", fields, self.api_keys["answers"])
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=[
                "answer_id", "question_id", "user_id", "display_name", "reputation", "is_accepted", "score"
            ])
            writer.writeheader()
            writer.writerows(answers)
        print(f"✅ Answers saved to {filepath}")


if __name__ == "__main__":
    # 🚨 Replace these placeholders with your actual API keys from StackExchange
    api_keys = {
        "users": "YOUR_USERS_API_KEY",
        "questions": "YOUR_QUESTIONS_API_KEY",
        "answers": "YOUR_ANSWERS_API_KEY"
    }

    # Output to 'data' directory in the parent directory
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_data_dir = os.path.join(parent_dir, "data")

    fetcher = StackOverflowDataFetcher(api_keys, output_dir=output_data_dir)

    # Download and save all datasets
    fetcher.save_users_csv()
    fetcher.save_questions_csv()
    fetcher.save_answers_csv()
