from database import get_db
import model


def format_smart_question(trait: str) -> str:
    # 1. Clean the trait string
    clean = trait.lower().strip().replace("-", "_")

    # 2. Manual overrides for words that need special grammar
    custom_overrides = {
        "fly": "Can your character fly?",
        "flying": "Can your character fly?",
        "magic": "Does your character use magic?",
        "superhero": "Is your character a superhero?",
        "villain": "Is your character a villain?",
        "human": "Is your character human?",
        "alien": "Is your character an alien?",
        "mutant": "Is your character a mutant?",
        "dead": "Is your character deceased?",
        "immortal": "Is your character immortal?",
        "rich": "Is your character wealthy?",
    }

    if clean in custom_overrides:
        return custom_overrides[clean]

    # 3. Rule-based patterns
    words = clean.split("_")
    first_word = words[0]
    rest = " ".join(words[1:]) if len(words) > 1 else ""

    # "wears_mask" -> "Does your character wear a mask?"
    # "uses_sword" -> "Does your character use a sword?"
    # "has_cape"   -> "Does your character have a cape?"
    if first_word.endswith("s") and first_word not in [
        "is",
        "has",
        "uses",
        "wears",
    ]:
        base_verb = first_word[:-1]
        return f"Does your character {base_verb} {rest}?".strip()

    if first_word in ["wears", "uses", "has", "possesses", "wields"]:
        base_verb = {
            "wears": "wear",
            "uses": "use",
            "has": "have",
            "possesses": "possess",
            "wields": "wield",
        }[first_word]
        return f"Does your character {base_verb} {rest}?".strip()

    # "can_teleport" -> "Can your character teleport?"
    if first_word == "can":
        return f"Can your character {rest}?".strip()

    # "is_female" -> "Is your character female?"
    if first_word == "is":
        return f"Is your character {rest}?".strip()

    # Fallback to a cleaner default than "related to"
    return f"Does your character have or use {clean.replace('_', ' ')}?"


def run():
    db = next(get_db())
    try:
        questions = db.query(model.Question).all()
        for q in questions:
            old_text = q.question
            new_text = format_smart_question(q.trait_key)
            q.question = new_text
            print(f"[{q.trait_key}] -> {new_text}")

        db.commit()
        print("\nAll 30 questions updated automatically!")
    finally:
        db.close()


if __name__ == "__main__":
    run()