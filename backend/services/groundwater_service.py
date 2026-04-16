import json
import os


data_path = os.path.join(os.path.dirname(__file__), "..", "data", "groundwater_ap.json")

try:
    with open(data_path, "r", encoding="utf-8") as f:
        groundwater_data = json.load(f)
except Exception:
    groundwater_data = {}


def get_groundwater_level(lat=None, lon=None):
    # TEMP: fallback logic (later can map properly)
    return "medium"
