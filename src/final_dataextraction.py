import requests
import csv
import time

# API settings - replace with your API keys
API_KEY_USERS = "rl_8RWSBZ7BCarzReCgVMg9cDzpt"
API_KEY_QUESTIONS = "rl_uBmXeYoeH3oLcKYDNYDCudX5M"
API_KEY_ANSWERS = "rl_nFKdDAUjifMojMpuMFrwMXjQg"
SITE = "stackoverflow"
PAGE_SIZE = 100  # Maximum allowed per request
OUTPUT_FILE_USERS = "stackoverflow_users.csv"
OUTPUT_FILE_QUESTIONS = "stackoverflow_questions.csv"
OUTPUT_FILE_ANSWERS = "stackoverflow_answers.csv"
BASE_URL = "https://api.stackexchange.com/2.3/"
HEADERS = {"Accept-Encoding": "gzip"}
PARAMS = {
    "site": SITE,
    "pagesize": PAGE_SIZE
}


# Function to fetch data with backoff handling
def fetch_data(endpoint, extract_fields, api_key):
    data = []
    page = 1
    while True:
        print(f"Fetching page {page} for {endpoint}...")
        response = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS,
                                params={**PARAMS, "page": page, "key": api_key})

        if response.status_code != 200:
            error_data = response.json()
            print("Error:", error_data)
            if "backoff" in error_data:
                backoff_time = error_data["backoff"]
                print(f"Backing off for {backoff_time} seconds...")
                time.sleep(backoff_time)
                continue
            break

        items = response.json().get("items", [])
        if not items:
            break

        for item in items:
            extracted_data = {field: item.get(field, "") for field in extract_fields}
            if "owner" in item:
                extracted_data.update({
                    "user_id": item["owner"].get("user_id", ""),
                    "display_name": item["owner"].get("display_name", ""),
                    "reputation": item["owner"].get("reputation", 0)
                })
            data.append(extracted_data)

        if not response.json().get("has_more"):
            break

        page += 1
        time.sleep(1)  # Avoid rate limiting

    return data


# Fetch and save users
def fetch_all_users():
    users = []
    page = 1
    while True:
        print(f"Fetching page {page} for user data")
        response = requests.get(f"{BASE_URL}users", headers=HEADERS,
                                params={**PARAMS, "page": page, "key": API_KEY_USERS})

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        data = response.json()
        users.extend(data.get("items", []))
        if not data.get("has_more", False):
            break  # Stop if no more data

        page += 1  # Respect API rate limits
        time.sleep(1)

    return users


users_data = fetch_all_users()
with open(OUTPUT_FILE_USERS, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["user_id", "display_name", "reputation", "gold_badges", "silver_badges", "bronze_badges"])
    for user in users_data:
        writer.writerow([
            user.get("user_id"),
            user.get("display_name"),
            user.get("reputation"),
            user.get("badge_counts", {}).get("gold", 0),
            user.get("badge_counts", {}).get("silver", 0),
            user.get("badge_counts", {}).get("bronze", 0)
        ])
print(f"Dataset saved as {OUTPUT_FILE_USERS}")

# Fetch and save questions
target_fields_questions = ["question_id", "tags", "score", "accepted_answer_id"]
questions = fetch_data("questions", target_fields_questions, API_KEY_QUESTIONS)
with open(OUTPUT_FILE_QUESTIONS, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["question_id", "tags", "score", "user_id", "reputation", "display_name",
                                              "accepted_answer_id"])
    writer.writeheader()
    writer.writerows(questions)
print(f"Dataset saved as {OUTPUT_FILE_QUESTIONS}")

# Fetch and save answers
target_fields_answers = ["answer_id", "question_id", "is_accepted", "score"]
answers = fetch_data("answers", target_fields_answers, API_KEY_ANSWERS)
with open(OUTPUT_FILE_ANSWERS, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["answer_id", "question_id", "user_id", "display_name", "reputation",
                                              "is_accepted", "score"])
    writer.writeheader()
    writer.writerows(answers)
print(f"Dataset saved as {OUTPUT_FILE_ANSWERS}")
