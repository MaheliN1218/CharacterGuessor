from database import Base
from sqlalchemy import Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

class Character(Base):
    __tablename__ = 'character'

    id: Mapped[int]= mapped_column(Integer, primary_key=True)
    name : Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    traits: Mapped[list["CharacterTrait"]] = relationship(
        back_populates="character", cascade="all, delete-orphan"
    )

class Question(Base):
    __tablename__ = 'question'

    id: Mapped[int]= mapped_column(Integer, primary_key=True, index=True)
    question: Mapped[str] = mapped_column(String(500), unique=True, index=True, nullable=False)
    trait_key: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)

    charactr_traits: Mapped[list["CharacterTrait"]] = relationship( back_populates="question")

class CharacterTrait(Base):
    __tablename__ = 'character_trait'
    id: Mapped[int]= mapped_column(Integer, primary_key=True, index=True)
    character_id: Mapped[int]= mapped_column(ForeignKey('character.id'), nullable=False)
    question_id: Mapped[int]= mapped_column(ForeignKey('question.id'), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    character: Mapped["Character"] = relationship(back_populates="traits")
    question: Mapped["Question"] = relationship(
        back_populates="character_traits"
    )

    __table_args__ = (
        UniqueConstraint(
            "character_id", "question_id", name="uq_char_question"
        ),
    )




