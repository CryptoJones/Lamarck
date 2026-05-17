#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Generate the Tier 3 held-out test set deterministically.

Produces exactly 100 problems across 10 domain templates (10 each).
Each problem is a (natural-language prompt, JSON-Schema draft-07)
pair. Output goes to:

  src/lamarck/eval/tier3_holdout/json_mode_problems.jsonl

The generator is seeded so the output is byte-for-byte stable.
Re-running it MUST produce an identical JSONL - the corpus is
locked under v1 and re-generation drift would break score
comparability across generations.

Run with no arguments from the repo root:
  python scripts/eval/generate_tier3_holdout.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path


# v1-locked seed. Changing this is a v2 event.
SEED = 0x42ACAC1A
TARGET_COUNT_PER_TEMPLATE = 10  # 10 templates x 10 = 100 problems


# ---- Reusable lexica (kept deterministic via seeded sampling) -------------

FIRST_NAMES = [
    "Aaron", "Beatrice", "Caleb", "Daria", "Elias", "Faye", "Gareth", "Hana",
    "Iris", "Jamal", "Kira", "Lior", "Mira", "Nasir", "Olive", "Petra",
    "Quincy", "Rohan", "Sasha", "Tomas", "Uma", "Vivek", "Wren", "Xanthe",
    "Yusuf", "Zora",
]
LAST_NAMES = [
    "Adair", "Bishara", "Cho", "Damaris", "Elman", "Fischer", "Gao",
    "Halverson", "Iqbal", "Jurek", "Khoury", "Larsen", "Mahaffey", "Nash",
    "Okafor", "Patel", "Quigley", "Reyes", "Sato", "Thatcher", "Uvarov",
    "Vargas", "Wexler", "Xu", "Yamamoto", "Zegers",
]
COUNTRIES = [
    "United States", "Germany", "Japan", "Brazil", "Kenya", "France",
    "Australia", "Singapore", "Argentina", "Iceland", "Vietnam", "Egypt",
]
CITIES = [
    ("Lincoln",    "United States"),
    ("Berlin",     "Germany"),
    ("Kyoto",      "Japan"),
    ("Curitiba",   "Brazil"),
    ("Nairobi",    "Kenya"),
    ("Lyon",       "France"),
    ("Hobart",     "Australia"),
    ("Singapore",  "Singapore"),
    ("Mendoza",    "Argentina"),
    ("Reykjavik",  "Iceland"),
    ("Hue",        "Vietnam"),
    ("Alexandria", "Egypt"),
]
DEVICE_IDS = [f"DEV-{1000 + i:04d}" for i in range(30)]
WORDS = [
    "ephemeral", "petrichor", "limerent", "sonder", "vellichor", "saudade",
    "yugen", "wabi-sabi", "fernweh", "hiraeth", "komorebi", "duende",
    "schadenfreude", "tarab", "iktsuarpok", "tsundoku",
]
POS_TAGS = ["noun", "adjective", "verb", "adverb"]
PRODUCT_KEYWORDS = [
    "Mechanical Keyboard", "Tea Kettle", "Hiking Boots", "Notebook",
    "Bluetooth Speaker", "Coffee Grinder", "Yoga Mat", "Wool Blanket",
    "Cast Iron Skillet", "Desk Lamp", "Bicycle Helmet", "Camera Strap",
]
PRODUCT_CATEGORIES = ["electronics", "kitchen", "outdoor", "office", "fitness"]
SERVICES = ["auth-svc", "billing", "search-index", "image-proc", "api-gateway",
            "metrics-shipper", "cache", "scheduler", "notifier", "queue"]
SERVICE_ENVS = ["dev", "staging", "production"]
COMPONENTS = ["database", "scheduler", "cache", "queue", "load-balancer",
              "auth-svc", "billing", "object-store", "dns", "cdn"]
STATUSES = ["healthy", "degraded", "down", "maintenance"]


# ---- Template builders -----------------------------------------------------

def gen_user_profile(rng: random.Random) -> dict:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    country = rng.choice(COUNTRIES)
    age_hint = rng.randint(18, 72)
    return {
        "input": (
            f"Create a user profile JSON object for {first} {last}, "
            f"age {age_hint}, living in {country}. Include their email "
            f"in the form firstname.lastname@example.com. Required keys: "
            f"name, email, age, country."
        ),
        "schema": {
            "type": "object",
            "required": ["name", "email", "age", "country"],
            "properties": {
                "name":    {"type": "string", "minLength": 1},
                "email":   {"type": "string", "format": "email"},
                "age":     {"type": "integer", "minimum": 0, "maximum": 150},
                "country": {"type": "string"},
            },
            "additionalProperties": True,
        },
    }


def gen_sensor_reading(rng: random.Random) -> dict:
    device = rng.choice(DEVICE_IDS)
    metric = rng.choice(["temperature", "pressure", "humidity", "light"])
    unit = {"temperature": "celsius", "pressure": "kPa",
            "humidity": "percent", "light": "lux"}[metric]
    value_hint = round(rng.uniform(0, 1000), 2)
    return {
        "input": (
            f"Emit a sensor reading JSON for device {device}: "
            f"a {metric} reading of approximately {value_hint} {unit}. "
            f"Include an ISO-8601 timestamp. Required keys: device_id, "
            f"metric, value, unit, timestamp."
        ),
        "schema": {
            "type": "object",
            "required": ["device_id", "metric", "value", "unit", "timestamp"],
            "properties": {
                "device_id": {"type": "string", "pattern": r"^DEV-\d{4}$"},
                "metric":    {"type": "string",
                              "enum": ["temperature", "pressure",
                                       "humidity", "light"]},
                "value":     {"type": "number"},
                "unit":      {"type": "string",
                              "enum": ["celsius", "kPa", "percent", "lux"]},
                "timestamp": {"type": "string"},
            },
            "additionalProperties": False,
        },
    }


def gen_dictionary_definition(rng: random.Random) -> dict:
    word = rng.choice(WORDS)
    hint_pos = rng.choice(POS_TAGS)
    n_examples = rng.randint(2, 4)
    return {
        "input": (
            f"Produce a dictionary-entry JSON for the word '{word}' "
            f"(give it as {hint_pos} if plausible, otherwise use the "
            f"correct part of speech). Include {n_examples} example "
            f"sentences. Required keys: word, part_of_speech, "
            f"definition, examples (array of strings)."
        ),
        "schema": {
            "type": "object",
            "required": ["word", "part_of_speech", "definition", "examples"],
            "properties": {
                "word":           {"type": "string"},
                "part_of_speech": {"type": "string", "enum": POS_TAGS},
                "definition":     {"type": "string", "minLength": 8},
                "examples":       {"type": "array",
                                   "items":    {"type": "string"},
                                   "minItems": 2},
            },
        },
    }


def gen_product_listing(rng: random.Random) -> dict:
    kw = rng.choice(PRODUCT_KEYWORDS)
    sku_n = rng.randint(10000, 99999)
    price = round(rng.uniform(9.99, 499.99), 2)
    return {
        "input": (
            f"Produce a product listing JSON for a '{kw}' (SKU prefix 'P-', "
            f"price near ${price}). Required keys: sku, name, price, "
            f"in_stock, category. The category must be one of: "
            f"{', '.join(PRODUCT_CATEGORIES)}. Pick the most appropriate."
        ),
        "schema": {
            "type": "object",
            "required": ["sku", "name", "price", "in_stock", "category"],
            "properties": {
                "sku":      {"type": "string", "pattern": r"^P-\d+$"},
                "name":     {"type": "string"},
                "price":    {"type": "number", "minimum": 0},
                "in_stock": {"type": "boolean"},
                "category": {"type": "string", "enum": PRODUCT_CATEGORIES},
            },
        },
    }


def gen_calendar_event(rng: random.Random) -> dict:
    title_root = rng.choice(["Sprint planning", "Design review",
                             "Customer call", "Retrospective", "1:1 meeting",
                             "Architecture sync", "Hiring panel",
                             "Demo day", "Onboarding session", "Standup"])
    n_attendees = rng.randint(2, 5)
    duration_min = rng.choice([15, 30, 45, 60, 90])
    return {
        "input": (
            f"Create a calendar event JSON titled '{title_root}' "
            f"({duration_min} minutes) with {n_attendees} attendees. "
            f"Required keys: title, start, end (both ISO-8601), "
            f"attendees (array of email addresses)."
        ),
        "schema": {
            "type": "object",
            "required": ["title", "start", "end", "attendees"],
            "properties": {
                "title": {"type": "string", "minLength": 1},
                "start": {"type": "string"},
                "end":   {"type": "string"},
                "attendees": {
                    "type":     "array",
                    "items":    {"type": "string", "format": "email"},
                    "minItems": 1,
                },
            },
        },
    }


def gen_geocoded_location(rng: random.Random) -> dict:
    city, country = rng.choice(CITIES)
    decimals = rng.choice([3, 4, 5])
    return {
        "input": (
            f"Produce a geocoded-location JSON for the city of {city}, "
            f"{country}. Round latitude and longitude to {decimals} "
            f"decimal places. Required keys: city, country, latitude, "
            f"longitude, population. Population must be a positive integer."
        ),
        "schema": {
            "type": "object",
            "required": ["city", "country", "latitude", "longitude",
                         "population"],
            "properties": {
                "city":       {"type": "string"},
                "country":    {"type": "string"},
                "latitude":   {"type": "number", "minimum": -90,
                               "maximum": 90},
                "longitude":  {"type": "number", "minimum": -180,
                               "maximum": 180},
                "population": {"type": "integer", "minimum": 1},
            },
        },
    }


def gen_config_block(rng: random.Random) -> dict:
    svc = rng.choice(SERVICES)
    env_hint = rng.choice(SERVICE_ENVS)
    port_hint = rng.randint(1024, 65535)
    return {
        "input": (
            f"Generate a service-configuration JSON block for '{svc}' "
            f"running in the '{env_hint}' environment on port ~{port_hint}. "
            f"Required keys: service_name, port (integer 1024-65535), "
            f"debug (boolean), env (one of dev/staging/production)."
        ),
        "schema": {
            "type": "object",
            "required": ["service_name", "port", "debug", "env"],
            "properties": {
                "service_name": {"type": "string", "minLength": 1},
                "port":         {"type": "integer", "minimum": 1024,
                                 "maximum": 65535},
                "debug":        {"type": "boolean"},
                "env":          {"type": "string", "enum": SERVICE_ENVS},
            },
            "additionalProperties": False,
        },
    }


def gen_recipe(rng: random.Random) -> dict:
    dish = rng.choice(["Spaghetti carbonara", "Miso soup", "Beef tagine",
                       "Mushroom risotto", "Banana bread", "Pad thai",
                       "Chana masala", "Apple crumble", "Pho ga",
                       "Borscht", "Shakshuka", "Pelmeni", "Bibimbap",
                       "Ratatouille", "Tom kha gai"])
    prep = rng.choice([15, 20, 30, 45, 60, 75, 90])
    servings = rng.choice([2, 3, 4, 6, 8])
    return {
        "input": (
            f"Produce a recipe JSON for '{dish}' that serves {servings} "
            f"with prep time around {prep} minutes. Required keys: dish, "
            f"prep_minutes, servings, ingredients. ingredients is an "
            f"array of objects each having 'name' and 'quantity'. At "
            f"least 3 ingredients."
        ),
        "schema": {
            "type": "object",
            "required": ["dish", "prep_minutes", "servings", "ingredients"],
            "properties": {
                "dish":         {"type": "string"},
                "prep_minutes": {"type": "integer", "minimum": 1},
                "servings":     {"type": "integer", "minimum": 1},
                "ingredients": {
                    "type": "array",
                    "minItems": 3,
                    "items": {
                        "type": "object",
                        "required": ["name", "quantity"],
                        "properties": {
                            "name":     {"type": "string"},
                            "quantity": {"type": "string"},
                        },
                    },
                },
            },
        },
    }


def gen_contact(rng: random.Random) -> dict:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    n_phones = rng.randint(1, 3)
    return {
        "input": (
            f"Produce a contact-card JSON for {first} {last} including "
            f"{n_phones} phone number(s) and at least one address with "
            f"street + city. Required keys: full_name, phones (array of "
            f"strings, at least 1), addresses (array of objects with "
            f"street + city, at least 1)."
        ),
        "schema": {
            "type": "object",
            "required": ["full_name", "phones", "addresses"],
            "properties": {
                "full_name": {"type": "string"},
                "phones":    {"type": "array",
                              "items":    {"type": "string"},
                              "minItems": 1},
                "addresses": {
                    "type":     "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["street", "city"],
                        "properties": {
                            "street": {"type": "string"},
                            "city":   {"type": "string"},
                        },
                    },
                },
            },
        },
    }


def gen_system_status(rng: random.Random) -> dict:
    component = rng.choice(COMPONENTS)
    status_hint = rng.choice(STATUSES)
    uptime = rng.randint(0, 30 * 24 * 3600)
    return {
        "input": (
            f"Emit a system-status JSON for component '{component}' "
            f"(currently '{status_hint}', uptime ~{uptime}s). Required "
            f"keys: component, status, uptime_seconds. status is one "
            f"of: healthy/degraded/down/maintenance. Include an optional "
            f"'message' string with a brief human-readable note."
        ),
        "schema": {
            "type": "object",
            "required": ["component", "status", "uptime_seconds"],
            "properties": {
                "component":      {"type": "string", "enum": COMPONENTS},
                "status":         {"type": "string", "enum": STATUSES},
                "uptime_seconds": {"type": "integer", "minimum": 0},
                "message":        {"type": "string"},
            },
        },
    }


TEMPLATES = (
    gen_user_profile, gen_sensor_reading, gen_dictionary_definition,
    gen_product_listing, gen_calendar_event, gen_geocoded_location,
    gen_config_block, gen_recipe, gen_contact, gen_system_status,
)


# ---- Driver -----------------------------------------------------------------

def _output_path() -> Path:
    """Resolve the output JSONL location relative to the repo root."""
    return (Path(__file__).resolve().parent.parent.parent
            / "src" / "lamarck" / "eval" / "tier3_holdout"
            / "json_mode_problems.jsonl")


def generate(out: Path | None = None) -> Path:
    """Generate the held-out set; return the output path.

    Re-runs are byte-identical because every template draws from a
    SEED-derived ``random.Random`` instance and no per-row IO is
    involved. The output path is overwritten if it already exists.

    Each prompt is suffixed with a stable case-id (e.g. "[case 042]")
    that guarantees uniqueness even when two rows draw the same
    randomly-sampled parameters - the small per-template parameter
    pools would otherwise produce duplicate prompts within the 10
    picks per template. The case-id is auxiliary metadata: it's part
    of the prompt the student sees and is distinct per row, so the
    held-out signal stays honest.
    """
    rng = random.Random(SEED)
    rows: list[dict] = []
    case_id = 0
    for template in TEMPLATES:
        for _ in range(TARGET_COUNT_PER_TEMPLATE):
            case_id += 1
            row = template(rng)
            row["input"] = f"{row['input']} [case {case_id:03d}]"
            rows.append(row)
    target = out if out is not None else _output_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


if __name__ == "__main__":
    path = generate()
    print(f"wrote {sum(1 for _ in path.open())} problems to {path}")
