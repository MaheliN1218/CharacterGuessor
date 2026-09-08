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

class GameSession:
    def __init__(self):
        self.num_characters = len(dataset.names)
        self.probabilities = np.ones(self.num_characters, dtype=float) / self.num_characters
        self.asked_indices = set()
        self.current_q_idx: Optional[int] = None
        self.question_count = 0
        self.history = []
        self.confidence_threshold = 0.65
        self.max_turns = 20


sessions: Dict[str, GameSession] = {}


def select_best_question(matrix: np.ndarray, probabilities: np.ndarray, asked_indices: set) -> Optional[int]:
    best_idx = None
    min_dist = float("inf")
    for q_idx in range(matrix.shape[1]):
        if q_idx in asked_indices:
            continue
        p_yes = np.sum(probabilities * matrix[:, q_idx])
        dist = abs(p_yes - 0.5)
        if dist < min_dist:
            min_dist = dist
            best_idx = q_idx
    return best_idx


def update_beliefs(probabilities: np.ndarray, matrix: np.ndarray, question_idx: int, answer_weight: float) -> np.ndarray:
    if answer_weight == 0.5:
        return probabilities
    features = matrix[:, question_idx]
    likelihood = 1.0 - abs(features - answer_weight)
    likelihood = np.clip(likelihood, 0.05, 0.95)
    posterior = probabilities * likelihood
    total = np.sum(posterior)
    return posterior / total if total > 0 else probabilities


def handling_same_category_features(session: GameSession, answered_feature: str, weight: float):
    if weight < 0.8:
        return

    for _, members in MUTUALLY_EXCLUSIVE_CATEGORIES.items():
        if answered_feature in members:
            similars = [f for f in members if f != answered_feature]
            for similar in similars:
                if similar in dataset.features:
                    sim_idx = dataset.features.index(similar)
                    session.asked_indices.add(sim_idx)
                    session.probabilities = update_beliefs(session.probabilities,
                        dataset.matrix,
                        sim_idx,
                        answer_weight=0.0,
                    )


class AnswerRequest(BaseModel):
    session_id: str
    choice: int


class RejectGuessRequest(BaseModel):
    session_id: str


class LearnRequest(BaseModel):
    session_id: str
    correct_name: str

@app.post("/start")
def start_game():
    session_id = str(uuid.uuid4())
    sessions[session_id] = GameSession()
    session = sessions[session_id]

    q_idx = select_best_question(dataset.matrix, session.probabilities, session.asked_indices)
    if q_idx is None:
        raise HTTPException(status_code=500, detail="Cannot initialize question tree.")

    session.asked_indices.add(q_idx)
    session.current_q_idx = q_idx
    session.question_count += 1
    feat_name = dataset.features[q_idx]

    return {
        "session_id": session_id,
        "question_number": session.question_count,
        "feature": feat_name,
        "question_text": f"Is your character {DISPLAY_NAMES.get(feat_name, feat_name)}?",
        "total_characters": session.num_characters,
    }
@app.post("/answer")
def submit_answer(req: AnswerRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session expired.")

    session = sessions[req.session_id]
    weight = CHOICE_WEIGHTS.get(req.choice, 0.5)

    last_q_idx = session.current_q_idx
    feat_name = dataset.features[last_q_idx]
    session.history.append((feat_name, weight))

    session.probabilities = update_beliefs(session.probabilities, dataset.matrix, last_q_idx, weight)
    handling_same_category_features(session, feat_name, weight)

    leader_idx = int(np.argmax(session.probabilities))
    leader_conf = float(session.probabilities[leader_idx])
    leader_name = str(dataset.names[leader_idx])

    if leader_conf >= session.confidence_threshold or session.question_count >= session.max_turns:
        return {
            "is_guess": True,
            "name": leader_name,
            "confidence": round(leader_conf * 100, 1),
            "question_number": session.question_count,
        }

    q_idx = select_best_question(dataset.matrix, session.probabilities, session.asked_indices)
    if q_idx is None:
        return {
            "is_guess": True,
            "name": leader_name,
            "confidence": round(leader_conf * 100, 1),
            "question_number": session.question_count,
        }

    session.asked_indices.add(q_idx)
    session.current_q_idx = q_idx
    session.question_count += 1
    next_feat = dataset.features[q_idx]

    return {
        "is_guess": False,
        "question_number": session.question_count,
        "feature": next_feat,
        "question_text": f"Is your character {DISPLAY_NAMES.get(next_feat, next_feat)}?",
        "top_candidate": leader_name,
        "top_confidence": round(leader_conf * 100, 1),
    }


@app.post("/reject-guess")
def reject_guess(req: RejectGuessRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[req.session_id]
    leader_idx = int(np.argmax(session.probabilities))

    session.probabilities[leader_idx] = 0.0
    total = np.sum(session.probabilities)
    if total > 0:
        session.probabilities /= total

    q_idx = select_best_question(dataset.matrix, session.probabilities, session.asked_indices)
    if q_idx is None:
        next_leader = int(np.argmax(session.probabilities))
        return {
            "is_guess": True,
            "name": str(dataset.names[next_leader]),
            "confidence": round(session.probabilities[next_leader] * 100, 1),
            "question_number": session.question_count,
        }

    session.asked_indices.add(q_idx)
    session.current_q_idx = q_idx
    session.question_count += 1
    next_feat = dataset.features[q_idx]

    return {
        "is_guess": False,
        "question_number": session.question_count,
        "feature": next_feat,
        "question_text": f"Is your character {DISPLAY_NAMES.get(next_feat, next_feat)}?",
    }


@app.post("/learn")
def teach_engine(req: LearnRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[req.session_id]
    features_dict = {}

    for feat, weight in session.history:
        if weight >= 0.8:
            features_dict[feat] = 1.0
        elif weight <= 0.2:
            features_dict[feat] = 0.0

    dataset.add_character(req.correct_name.strip(), features_dict)
    del sessions[req.session_id]

    return {
        "message": f"Successfully incorporated '{req.correct_name}' into character repository.",
        "new_pool_size": len(dataset.names),
    }


if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_ui():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    elif os.path.exists("index.html"):
        return FileResponse("index.html")
    raise HTTPException(status_code=404, detail="index.html not found.")

