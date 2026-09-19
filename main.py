import uuid
from contextlib import asynccontextmanager

from database import Base, engine, get_db
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import model
import numpy as np
from pydantic import BaseModel
from sqlalchemy.orm import Session

Base.metadata.create_all(bind=engine)

GLOBAL_STATE = {
    "character_names": [],
    "questions": [],
    "matrix": np.empty((0, 0)),
}


SESSIONS = {}


def reload_matrix_from_db(db: Session):

    chars = db.query(model.Character).order_by(model.Character.id).all()
    questions = db.query(model.Question).order_by(model.Question.id).all()

    char_names = [c.name for c in chars]
    char_id_to_idx = {c.id: idx for idx, c in enumerate(chars)}
    q_id_to_idx = {q.id: idx for idx, q in enumerate(questions)}

    matrix = np.zeros((len(chars), len(questions)), dtype=float)

    traits = db.query(model.CharacterTrait).all()
    for t in traits:
        if t.character_id in char_id_to_idx and t.question_id in q_id_to_idx:
            row = char_id_to_idx[t.character_id]
            col = q_id_to_idx[t.question_id]
            matrix[row, col] = t.value

    GLOBAL_STATE["character_names"] = char_names
    GLOBAL_STATE["questions"] = [
        {"id": q.id, "text": q.question, "trait_key": q.trait_key}
        for q in questions
    ]
    GLOBAL_STATE["matrix"] = matrix
    print(f"Matrix loaded: {len(char_names)} characters, {len(questions)} questions.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = next(get_db())
    try:
        reload_matrix_from_db(db)
    finally:
        db.close()
    yield
    SESSIONS.clear()


app = FastAPI(title="Character Guesser Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def calculate_entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0


def select_best_question(probs: np.ndarray, asked_q_indices: set[int]) -> int | None:
    matrix = GLOBAL_STATE["matrix"]
    num_questions = matrix.shape[1]

    available_indices = [i for i in range(num_questions) if i not in asked_q_indices]
    if not available_indices:
        return None

    current_entropy = calculate_entropy(probs)
    best_gain = -1.0
    best_q_idx = available_indices[0]

    for q_idx in available_indices:
        q_column = matrix[:, q_idx]

        p_yes = np.sum(probs * q_column)
        p_no = 1.0 - p_yes

        if p_yes <= 0.001 or p_no <= 0.001:
            continue

        probs_given_yes = (probs * q_column) / p_yes
        probs_given_no = (probs * (1.0 - q_column)) / p_no

        expected_entropy = (p_yes * calculate_entropy(probs_given_yes)) + (
            p_no * calculate_entropy(probs_given_no)
        )
        info_gain = current_entropy - expected_entropy

        if info_gain > best_gain:
            best_gain = info_gain
            best_q_idx = q_idx

    return best_q_idx


class AnswerRequest(BaseModel):
    session_id: str
    choice: int  # 1: Yes, 2: Probably, 3: Don't Know, 4: Probably Not, 5: No


class RejectGuessRequest(BaseModel):
    session_id: str


class LearnRequest(BaseModel):
    session_id: str
    correct_name: str



from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@app.get("/")
def serve_index():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/start")
def start_game():
    if len(GLOBAL_STATE["character_names"]) == 0:
        raise HTTPException(
            status_code=400,
            detail="Database has no characters. Please seed your data first.",
        )

    session_id = str(uuid.uuid4())
    num_chars = len(GLOBAL_STATE["character_names"])
    initial_probs = np.full(num_chars, 1.0 / num_chars, dtype=float)

    first_q_idx = select_best_question(initial_probs, asked_q_indices=set())
    if first_q_idx is None:
        first_q_idx = 0

    SESSIONS[session_id] = {
        "probs": initial_probs,
        "asked_q_indices": [first_q_idx],
        "question_count": 1,
        "rejected_characters": [],
        "last_guess": None,
    }

    q_data = GLOBAL_STATE["questions"][first_q_idx]
    return {
        "session_id": session_id,
        "question_number": 1,
        "question_id": q_data["id"],
        "question_text": q_data["text"],
    }


@app.post("/answer")
def submit_answer(req: AnswerRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = SESSIONS[req.session_id]
    current_q_idx = session["asked_q_indices"][-1]
    probs = session["probs"]
    matrix = GLOBAL_STATE["matrix"]

    q_column = matrix[:, current_q_idx]

    # Fuzzy belief mapping
    weight_map = {
        1: 0.95,  # Yes
        2: 0.75,  # Probably
        3: 0.50,  # Don't Know
        4: 0.25,  # Probably Not
        5: 0.05,  # No
    }
    user_weight = weight_map.get(req.choice, 0.50)

    likelihood = (q_column * user_weight) + ((1.0 - q_column) * (1.0 - user_weight))
    unnormalized = probs * likelihood
    total = np.sum(unnormalized)

    if total > 0:
        probs = unnormalized / total
    session["probs"] = probs

    sorted_indices = np.argsort(probs)[::-1]
    top_idx = sorted_indices[0]
    top_prob = float(probs[top_idx])
    top_char_name = GLOBAL_STATE["character_names"][top_idx]

    if top_prob >= 0.75 or session["question_count"] >= 20:
        session["last_guess"] = top_char_name
        return {
            "is_guess": True,
            "name": top_char_name,
            "confidence": round(top_prob * 100, 1),
        }

    next_q_idx = select_best_question(
        probs, asked_q_indices=set(session["asked_q_indices"])
    )

    if next_q_idx is None:
        session["last_guess"] = top_char_name
        return {
            "is_guess": True,
            "name": top_char_name,
            "confidence": round(top_prob * 100, 1),
        }

    session["asked_q_indices"].append(next_q_idx)
    session["question_count"] += 1

    next_q = GLOBAL_STATE["questions"][next_q_idx]
    return {
        "is_guess": False,
        "question_number": session["question_count"],
        "question_id": next_q["id"],
        "question_text": next_q["text"],
        "top_candidate": top_char_name,
        "top_confidence": round(top_prob * 100, 1),
    }


@app.post("/reject-guess")
def reject_guess(req: RejectGuessRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = SESSIONS[req.session_id]
    rejected_name = session.get("last_guess")

    if rejected_name in GLOBAL_STATE["character_names"]:
        char_idx = GLOBAL_STATE["character_names"].index(rejected_name)
        session["probs"][char_idx] = 0.0

        total = np.sum(session["probs"])
        if total > 0:
            session["probs"] /= total

    session["rejected_characters"].append(rejected_name)

    if np.max(session["probs"]) <= 0.001 or session["question_count"] >= 25:
        return {"can_learn": True}

    next_q_idx = select_best_question(
        session["probs"], asked_q_indices=set(session["asked_q_indices"])
    )

    if next_q_idx is None:
        return {"can_learn": True}

    session["asked_q_indices"].append(next_q_idx)
    session["question_count"] += 1

    next_q = GLOBAL_STATE["questions"][next_q_idx]
    return {
        "can_learn": False,
        "is_guess": False,
        "question_number": session["question_count"],
        "question_text": next_q["text"],
    }


@app.post("/learn")
def learn_character(req: LearnRequest, db: Session = Depends(get_db)):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session expired.")

    session = SESSIONS[req.session_id]
    cleaned_name = req.correct_name.strip()

    char_record = (
        db.query(model.Character)
        .filter(model.Character.name.ilike(cleaned_name))
        .first()
    )
    if not char_record:
        char_record = model.Character(name=cleaned_name)
        db.add(char_record)
        db.flush()

    for q_idx in session["asked_q_indices"]:
        q_meta = GLOBAL_STATE["questions"][q_idx]
        existing_trait = (
            db.query(model.CharacterTrait)
            .filter_by(character_id=char_record.id, question_id=q_meta["id"])
            .first()
        )
        if not existing_trait:
            db.add(
                model.CharacterTrait(
                    character_id=char_record.id,
                    question_id=q_meta["id"],
                    value=0.5,
                )
            )

    db.commit()
    reload_matrix_from_db(db)

    return {
        "status": "success",
        "message": f"Successfully registered {cleaned_name}.",
    }
