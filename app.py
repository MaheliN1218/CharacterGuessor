import os
import uuid
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Character guessing Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_PATH = "character.csv"

CHOICE_WEIGHTS = {
    1: 1.0,  # Yes
    2: 0.8,  # Probably
    3: 0.5,  # Don't know
    4: 0.2,  # Probably not
    5: 0.0,  # No
}

DISPLAY_NAMES = {
    "is_marvel": "from the Marvel Universe",
    "is_dc": "from the DC Universe",
    "is_good": "a hero or aligned with good",
    "is_bad": "a villain or antagonist",
    "is_male": "male",
    "is_human": "a standard biological human without innate alien or mutant genetics",
    "is_mutant": "a mutant born with the X-gene",
    "is_alien_or_god": "an alien, deity, or mythological god",
    "is_cyborg_or_tech": "an android, cyborg, or artificial being",
    "hair_black": "someone with black hair",
    "hair_blond": "someone with blond hair",
    "hair_red_or_auburn": "someone with red or auburn hair",
    "is_bald": "completely bald or hairless",
    "eyes_blue": "someone with blue eyes",
    "eyes_red": "someone with red or glowing eyes",
    "skin_non_human": "someone with non-human skin (e.g. green, blue, metallic, or stone)",
    "is_tall_or_massive": "unusually tall or a giant (over 6'3\" / 190 cm)",
    "is_short": "shorter than average (under 5'8\" / 173 cm)",
    "power_magic_mystic": "a practitioner of sorcery, magic, or mystical arts",
    "power_tech_gadgets": "primarily reliant on high-tech gadgets, suits, or weaponry rather than innate biology",
    "power_psionic": "a telepath, telekinetic, or possessor of mental psionic powers",
    "power_cosmic": "a wielder of cosmic-level energy or reality manipulation",
    "is_pure_martial_artist": "primarily a non-powered hand-to-hand martial artist or street vigilante",
    "can_fly": "capable of self-propelled flight",
    "has_super_strength": "endowed with godlike or incalculable physical strength (75+ tons)",
    "has_healing_factor": "known for an accelerated regenerative healing factor",
    "uses_energy_blasts": "capable of projecting energy blasts, beams, or lightning",
    "wears_powered_armor": "wearing a full set of powered or technological battle armor",
    "wields_bladed_weapon": "known for using swords, claws, or bladed weapons in combat",
    "is_genius_intellect": "considered a genius or super-genius intellect",
}

MUTUALLY_EXCLUSIVE_CATEGORIES = {
    "publisher": ["is_marvel", "is_dc"],
    "alignment": ["is_good", "is_bad"],
    "hair_color": ["hair_black", "hair_blond", "hair_red_or_auburn", "is_bald"],
    "height_class": ["is_tall_or_massive", "is_short"],
    "eye_color": ["eyes_blue", "eyes_red"],
    "species_origin": ["is_human", "is_mutant", "is_alien_or_god", "is_cyborg_or_tech"],
}



class DatasetManager:
    def __init__(self, path: str):
        self.path = path
        self.load()

    def load(self):
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            raise FileNotFoundError(f"'{self.path}' is missing or empty. Please run generate_csv.py first.")
        self.df = pd.read_csv(self.path)
        self.names = self.df["name"].values
        self.features = [c for c in self.df.columns if c != "name"]
        self.matrix = self.df[self.features].astype(float).values

    def add_character(self, name: str, feature_dict: Dict[str, float]):
        new_row = {"name": name}
        for f in self.features:
            new_row[f] = feature_dict.get(f, 0.5)
        new_df = pd.DataFrame([new_row])
        self.df = pd.concat([self.df, new_df], ignore_index=True).drop_duplicates(subset=["name"])
        self.df.to_csv(self.path, index=False)
        self.load()


dataset = DatasetManager(CSV_PATH)
