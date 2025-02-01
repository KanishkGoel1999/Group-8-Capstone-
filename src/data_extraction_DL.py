import requests
import csv
import time

# API settings
API_KEY = "rl_4ZRmfRKVixxm8iCtGQYo9gDmk"
SITE = "stackoverflow"
BASE_URL = "https://api.stackexchange.com/2.3/"
HEADERS = {"Accept-Encoding": "gzip"}
PARAMS = {
    "order": "desc",
    "sort": "activity",
    "site": SITE,
    "pagesize": 100,
    "key": API_KEY
}


# Fetch data function
def fetch_data(endpoint):
    data = []
    page = 1
    while True:
        print(f"Fetching page {page} for {endpoint}...")
        response = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params={**PARAMS, "page": page})
        if response.status_code != 200:
            print("Error:", response.json())
            break

        items = response.json().get("items", [])
        if not items:
            break

        for item in items:
            user_id = item.get("owner", {}).get("user_id")
            data.append({
                "user_id": user_id,
                "post_type": "question" if endpoint == "questions" else "answer",
                "tags": ",".join(item.get("tags", [])),
                "score": item.get("score", 0),
                "comment_count": item.get("comment_count", 0),
                "reputation": item.get("owner", {}).get("reputation", 0),
                "badge_counts": item.get("owner", {}).get("badge_counts", {}),
                "view_count": item.get("view_count", 0)
            })

        if not response.json().get("has_more"):
            break

        page += 1
        time.sleep(1)  # Avoid rate limiting

    return data


# Fetch and save dataset
questions = fetch_data("questions")
answers = fetch_data("answers")

# Merge data
dataset = questions + answers

# Save to CSV
csv_file = "stackexchange_dataset.csv"
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["user_id", "post_type", "tags", "score", "comment_count", "reputation", "badge_counts", "view_count"])
    writer.writeheader()
    writer.writerows(dataset)

print(f"Dataset saved as {csv_file}")