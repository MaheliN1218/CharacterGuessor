# 🦸‍♂️ Superhero Akinator: Adaptive Bayesian Deduction Engine

An Akinator-style web application that deduces any Marvel or DC character through adaptive questioning, minimizes guesswork using Information Entropy, and learns unknown characters dynamically from player feedback.

## 🧾 Project Overview

Standard decision-tree systems use rigid, hardcoded question paths that break when a user is uncertain. This engine formulates deduction as a probabilistic search problem using:

- **Information Gain via Shannon Entropy:** Dynamically selects the next question that splits the remaining probability distribution most evenly, eliminating characters in the fewest steps.
- **Fuzzy Bayesian Belief Updating:** Supports nuanced answers (Yes, Probably, Don't Know, Probably Not, No) by adjusting likelihood vectors rather than hard-filtering candidates.
- **Dynamic Feedback & Learning Loop:** If the engine fails to deduce the character within 3 guesses, it prompts the player to teach it the identity, appending the character and trait vector directly to the database.

## 🌐 Live Demo

🎯 **Live Web App:** [Try the Character Guesser Here](https://characterguesser.vercel.app)

The interactive web dashboard lets you:
- Experience dynamic question routing based on real-time candidate entropy.
- Track confidence scores updating live after every response.
- Teach the system new characters via an automated fallback loop after 3 failed guesses.

## 🧮 Dataset & Trait Architecture

- **Characters:** 200+ curated Marvel and DC heroes, anti-heroes, and villains.
- **Trait Dimensions:** 30 binary and fuzzy feature columns covering:
  - **Universes & Alignments:** Marvel, DC, Hero/Good, Villain/Bad.
  - **Species & Biology:** Human, Mutant, Alien/Deity, Cyborg.
  - **Physical Characteristics:** Hair color, eye color, height/build, baldness, non-human skin.
  - **Powers & Archetypes:** Flight, super strength, healing factor, energy blasts, magic/mystic, tech gadgets, martial arts mastery.

## 🏗️ Repository Structure
```text
CharacterGuesser/
├── static/
│   └── index.html          # Clean dark-mode UI with confidence gauges
├── database.py             # SQLAlchemy session manager & PostgreSQL pool
├── main.py                 # FastAPI serverless backend & deduction algorithm
├── model.py                # Normalized schema (Character, Question, CharacterTrait)
├── mig.py                  # Seed and matrix normalization pipeline
├── updateQ.py              # Trait prompt standardization script
├── requirements.txt        # Production dependency specifications
├── vercel.json             # Vercel serverless build and routing manifest
├── .gitignore              # Ignored local environments and artifacts
└── README.md               # Technical project documentation
```
## 📋 What Was Built

### 1. Bayesian Probability Update Pipeline
Rather than binary filtering, each character maintains a prior probability. When a player submits an answer with a given weight, the likelihood is calculated across trait values:
- Computes unnormalized posterior probabilities based on the user's fuzzy response.
- Dynamically normalizes probabilities across all 200+ candidates after each step.

### 2. Shannon Entropy Question Selection
The engine evaluates all unasked questions to select the one that yields the maximum Expected Information Gain:
- Evaluates expected reduction in entropy across candidate splits.
- Discards non-informative traits dynamically during play.

### 3. Dynamic Self-Learning Loop
When 3 guesses fail:
- The engine yields to the player and opens a submission prompt.
- The player names the correct character.
- The server writes the character to Neon PostgreSQL and maps the answers given during the session directly into the character trait relationship table.

## 📊 Results & Performance

- **Average Questions to Deduction:** 10–14 questions for popular characters.
- **Deduction Threshold:** 70.0% confidence cutoff for triggering character guesses.
- **Failure Handling:** Enforces a hard limit of 3 incorrect guesses before engaging the learning interface.
- **Cloud Latency:** Sub-100ms response time on Vercel Serverless connected to Neon PostgreSQL.

## 🛠️ Tools & Technologies

- **Language:** Python 3.11
- **Backend Framework:** FastAPI, Uvicorn
- **Mathematical Computation:** NumPy
- **Database & ORM:** PostgreSQL (Neon Serverless), SQLAlchemy
- **Deployment Platform:** Vercel (Serverless Functions)
- **Frontend:** Responsive Vanilla HTML5, CSS3, JavaScript (Fetch API)

## ⚙️ How to Run Locally

### 1. Clone this repository
```text
git clone [https://github.com/MaheliN1218/CharacterGuessor.git](https://github.com/MaheliN1218/CharacterGuessor.git)
cd CharacterGuessor
```
### 2. Create and activate a virtual environment
```text
python -m venv .venv
```
Windows:
```text
.venv\Scripts\activate
```
Mac/Linux:
```text
source .venv/bin/activate
```

### 3. Install packages
```text
pip install -r requirements.txt
```
### 4. Configure Database and Seed
Set your PostgreSQL connection string:
```text
$env:DATABASE_URL="your-postgresql-connection-string"
```

Run migrations:
```text
python mig.py
```
### 5. Launch the app
```text
uvicorn main:app --reload
```
Open http://localhost:8000 in your browser.

Live site- https://character-guessor-8wky-6klomdb5f-mahelin1218.vercel.app/

## 📚 References & Background

- Shannon, C. E. (1948). A Mathematical Theory of Communication.
- Russell, S., & Norvig, P. Artificial Intelligence: A Modern Approach (Decision Trees & Probabilistic Inference).
- Akinator Engine Logic: Twenty Questions (20Q) heuristic algorithms and Bayesian neural models.
