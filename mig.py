import os
from database import Base, SessionLocal, engine
from model import Character, CharacterTrait, Question
import pandas as pd


def run_migration():
    print("Creating tables in PostgreSQL if they don't exist...")
    Base.metadata.create_all(bind=engine)

    csv_file = "character.csv"
    if not os.path.exists(csv_file):
        print(f"Error: Could not find '{csv_file}' in the current folder.")
        return

    df = pd.read_csv(csv_file)
    print(
        f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns successfully."
    )

    db = SessionLocal()
    try:
        # Determine the name column (either 'name' or the 1st column)
        char_col = "name" if "name" in df.columns else df.columns[0]
        trait_cols = [col for col in df.columns if col != char_col]

        # 1. Sync Question list
        question_map = {}
        for trait in trait_cols:
            clean_trait = trait.strip()
            q = (
                db.query(Question)
                .filter(Question.trait_key == clean_trait)
                .first()
            )
            if not q:
                readable_prompt = (
                    f"does your character {clean_trait.replace('_', ' ')}?"
                )
                q = Question(question=readable_prompt, trait_key=clean_trait)
                db.add(q)
                db.flush()
            question_map[clean_trait] = q.id

        # 2. Sync Characters and Trait Matrix
        for _, row in df.iterrows():
            char_name = str(row[char_col]).strip()
            if not char_name or char_name.lower() == "nan":
                continue

            char = (
                db.query(Character)
                .filter(Character.name == char_name)
                .first()
            )
            if not char:
                char = Character(name=char_name)
                db.add(char)
                db.flush()

            for trait in trait_cols:
                clean_trait = trait.strip()
                raw_val = row[trait]
                val = 1.0 if str(raw_val).strip() in ["1", "1.0", "True", "true"] else 0.0

                # Check if trait already exists
                existing = (
                    db.query(CharacterTrait)
                    .filter(
                        CharacterTrait.character_id == char.id,
                        CharacterTrait.question_id == question_map[clean_trait],
                    )
                    .first()
                )

                if not existing:
                    db.add(
                        CharacterTrait(
                            character_id=char.id,
                            question_id=question_map[clean_trait],
                            value=val,
                        )
                    )

        db.commit()
        print("Success: All data imported into PostgreSQL.")
    except Exception as err:
        db.rollback()
        print(f"Migration failed: {err}")
    finally:
        db.close()


if __name__ == "__main__":
    run_migration()