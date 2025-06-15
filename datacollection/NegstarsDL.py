import csv
import requests
import os

output_dir = "sdss_negatives"
os.makedirs(output_dir, exist_ok=True)

with open("data/GalaxyZoo1/NegStarsWithURL.csv", "r") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        url = row["url"]
        obj_id = row["objID"]
        response = requests.get(url)
        if response.status_code == 200:
            filename = f"stars_{obj_id}.jpg"
            with open(os.path.join(output_dir, filename), "wb") as img_file:
                img_file.write(response.content)
        if i % 100 == 0:
            print(f"Downloaded {i} images...")