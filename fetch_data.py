import os
import pandas as pd


RAW_INPUT = "marvel_dc_raw.csv" if os.path.exists("marvel_dc_raw.csv") else "character.csv"
OUTPUT_FILE = "character.csv"

raw_df = pd.read_csv(RAW_INPUT)


raw_df.columns = [c.strip().replace('\ufeff', '').lower() for c in raw_df.columns]

vectorized_rows = []

for _,row in raw_df.iterrows():
    def get_val(keys):
        for k in keys:
            if k in row and pd.notna(row[k]):
                return str(row[k]).strip().lower()

        return ""

    name= get_val(['character_name','name'])
    if not name:
        continue

    pub = get_val(["publisher"])
    align = get_val(["alignment"])
    gender = get_val(["gender"])
    species = get_val(["species"])
    hair = get_val(["hair_color", "hair"])
    eye = get_val(["eye_color", "eye"])
    skin = get_val(["skin_tone", "skin"])
    h_class = get_val(["height_class"])
    power_type = get_val(["primary_power_type", "power_type"])
    powers_desc = get_val(["key_powers_and_traits", "powers"])
    strength = get_val(["strength_tier", "strength"])
    intel = get_val(["intelligence_tier", "intel"])
    weapon = get_val(["signature_weapon_gear", "weapon", "gear"])

    vec = {

        "name": name,

        # Universe & Demographics
        "is_marvel": 1.0 if pub == "marvel" else 0.0,
        "is_dc": 1.0 if pub == "dc" else 0.0,
        "is_good": 1.0 if align == "good" else 0.0,
        "is_bad": 1.0 if align == "bad" else 0.0,
        "is_male": 1.0 if gender == "male" else 0.0,
        "is_human": 1.0 if "human" in species and "mutate" not in species and "alien" not in species else 0.0,
        "is_mutant": 1.0 if "mutant" in species else 0.0,
        "is_alien_or_god": 1.0 if any(
            k in species for k in ["alien", "god", "kryptonian", "martian", "asgardian", "new god"]) else 0.0,
        "is_cyborg_or_tech": 1.0 if any(k in species for k in ["android", "cyborg", "synthezoid"]) else 0.0,

        # Physical Appearance

        "hair_black": 1.0 if "black" in hair else 0.0,
        "hair_blond": 1.0 if "blond" in hair else 0.0,
        "hair_red_or_auburn": 1.0 if any(k in hair for k in ["red", "auburn"]) else 0.0,
        "is_bald": 1.0 if "bald" in hair or hair == "none" else 0.0,
        "eyes_blue": 1.0 if "blue" in eye else 0.0,
        "eyes_red": 1.0 if "red" in eye else 0.0,
        "skin_non_human": 1.0 if any(
            k in skin for k in ["green", "blue", "metal", "purple", "stone", "chalk", "gold"]) else 0.0,
        "is_tall_or_massive": 1.0 if h_class in ["tall", "massive"] else 0.0,
        "is_short": 1.0 if h_class == "short" else 0.0,

        # Powers

        "power_magic_mystic": 1.0 if "magic" in power_type else 0.0,
        "power_tech_gadgets": 1.0 if "tech" in power_type else 0.0,
        "power_psionic": 1.0 if "psionic" in power_type else 0.0,
        "power_cosmic": 1.0 if "cosmic" in power_type else 0.0,
        "is_pure_martial_artist": 1.0 if "martial" in power_type else 0.0,

        # Combat and Gear

        "can_fly": 1.0 if "flight" in powers_desc or "fly" in powers_desc else 0.0,
        "has_super_strength": 1.0 if strength in ["incalculable", "cosmic/godlike"] else 0.0,
        "has_healing_factor": 1.0 if any(k in powers_desc for k in ["healing", "regeneration"]) else 0.0,
        "uses_energy_blasts": 1.0 if any(
            k in powers_desc for k in ["blast", "beam", "laser", "projectile", "lightning", "energy"]) else 0.0,
        "wears_powered_armor": 1.0 if "armor" in weapon or "suit" in weapon else 0.0,
        "wields_bladed_weapon": 1.0 if any(
            k in weapon for k in ["sword", "katana", "claws", "blade", "daggers"]) else 0.0,
        "is_genius_intellect": 1.0 if "genius" in intel else 0.0,

    }

    vectorized_rows.append(vec)

df_clean = pd.DataFrame(vectorized_rows)
df_clean.to_csv(OUTPUT_FILE, index=False)

print(    f"Done! Generated '{OUTPUT_FILE}' with {len(df_clean)} characters and {len(df_clean.columns) - 1} binary feature columns.")

