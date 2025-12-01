#!/usr/bin/env python3

import sys
import json
import urllib.parse
import urllib.request
import pandas as pd
import os
import re
import time
from openai import OpenAI

SEVERITY_CALLS = 0
OPENAI_RPM_LIMIT = 500
_rate_window_start = time.time()
_rate_window_count = 0


def enforce_openai_rate_limit():
    global _rate_window_start, _rate_window_count
    now = time.time()
    elapsed = now - _rate_window_start
    if elapsed >= 60:
        _rate_window_start = now
        _rate_window_count = 0
        return
    if _rate_window_count >= OPENAI_RPM_LIMIT:
        time.sleep(60 - elapsed)
        _rate_window_start = time.time()
        _rate_window_count = 0
from dotenv import load_dotenv

load_dotenv() 


KNOWN_SEVERITY = {
    "underage drinking": 1,
    "liquor law violation": 1,
    "open container": 1,
    "noise complaint": 1,
    "vandalism": 2,
    "criminal damage": 2,
    "trespass": 2,
    "theft": 3,
    "retail theft": 3,
    "vehicle theft": 3,
    "burglary": 4,
    "robbery": 4,
    "aggravated battery": 5,
    "weapons offense": 5,
    "sexual assault": 5,
}

INPUT_PATH = "data/Clery Crime Log - Police Contacts Only - 2021-October 31 2025.xlsx"


def normalize_description(text):
    s = (text or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_severity_cache(path="severity_cache.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_severity_cache(cache, path="severity_cache.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_severity(description, api_key=None, cache_path="severity_cache.json"):
    global SEVERITY_CALLS
    SEVERITY_CALLS += 1
    try:
        if SEVERITY_CALLS % 100 == 0:
            print(f"severity {SEVERITY_CALLS}")
        else:
            sys.stdout.write("."); sys.stdout.flush()
    except Exception:
        pass

    norm = normalize_description(description)

    cache = load_severity_cache(cache_path)
    if norm in cache:
        return int(cache[norm])

    for key, val in KNOWN_SEVERITY.items():
        if key in norm:
            cache[norm] = int(val)
            save_severity_cache(cache, cache_path)
            return int(val)

    key = api_key or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    system = (
        "You are a strict crime severity classifier. Output only a single digit 1-5.\n"
        "Scale: 1=minor/administrative (e.g., underage drinking, noise), 2=low harm (trespass, vandalism), "
        "3=property crimes (most theft), 4=threat/force without severe injury (robbery, burglary), "
        "5=violent/sexual/weapons or severe harm (aggravated battery, sexual assault).\n"
        "Examples:\n"
        "- 'Underage drinking in dorm' -> 1\n"
        "- 'Noise complaint at apartment' -> 1\n"
        "- 'Criminal damage to property (vandalism)' -> 2\n"
        "- 'Theft of bicycle from rack' -> 3\n"
        "- 'Residential burglary, suspect not present' -> 4\n"
        "- 'Armed robbery at convenience store' -> 4\n"
        "- 'Aggravated battery with serious injury' -> 5\n"
        "- 'Reported sexual assault' -> 5\n"
    )
    user = f"Crime description: {description}\nSeverity (1-5) only:"
    enforce_openai_rate_limit()
    resp = client.responses.create(
        model="gpt-5-nano",
        input=f"{system}\n\n{user}"
    )
    # count the request for rate tracking
    global _rate_window_count
    _rate_window_count += 1
    sev = int(resp.output_text.strip())
    cache[norm] = sev
    save_severity_cache(cache, cache_path)
    return sev

def geocode_locations(df, address_column="Location", api_key=None):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
    lats, lons = [], []
    addrs = df[address_column].astype(str).tolist()
    total = len(addrs)
    print(f"Geocoding {total} addresses...")
    for i, address in enumerate(addrs, 1):
        # Remove 'University of Illinois' if present (case-insensitive), then tidy spaces/commas
        cleaned = re.sub(r"(?i)university of illinois", "", address)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" ,")
        params = {"address": cleaned + " Champaign", "key": key}
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        status = data.get("status")
        if status != "OK":
            raise RuntimeError(f"Geocoding failed: {status} - {data.get('error_message', '')}")
        loc = data["results"][0]["geometry"]["location"]
        lats.append(loc["lat"])
        lons.append(loc["lng"])
        if i % 100 == 0 or i == total:
            print(f"geocode {i}/{total}")
    df = df.copy()
    df["lat"] = lats
    df["lon"] = lons
    return df


def save_as_csv(df, output_path):
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    input_path = INPUT_PATH
    output_path = "data/crime_log_processed"
    # df = pd.read_excel(input_path)

    # df = geocode_locations(df, address_column="Location")
    # save_as_csv(df, output_path + "_location.csv")


    df = pd.read_csv("data/crime_log_processed_location.csv")
    df = df.dropna()
    df["severity"] = df["Description"].apply(get_severity)
    save_as_csv(df, output_path + "_full.csv")

    print(f"Saved: {output_path}")
