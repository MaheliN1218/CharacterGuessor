# 🦸‍♂️ Superhero Akinator: Adaptive Bayesian Deduction Engine

An Akinator-style web application that deduces any Marvel or DC character through adaptive questioning, minimizes guesswork using Information Entropy, and learns unknown characters dynamically from player feedback.

---

## 🧾 Project Overview

Standard decision-tree systems use rigid, hardcoded question paths that break when a user is uncertain. This engine formulates deduction as a **probabilistic search problem** using:

- **Information Gain via Shannon Entropy:** Dynamically selects the next question that splits the remaining probability distribution most evenly ($H(X) = -\sum P(x) \log_2 P(x)$), eliminating characters in the fewest steps.
- **Fuzzy Bayesian Belief Updating:** Supports nuanced answers (*Yes*, *Probably*, *Don't Know*, *Probably Not*, *No*) by adjusting likelihood vectors rather than hard-filtering candidates.
- **Dynamic Feedback & Learning Loop:** If the engine fails to deduce the character within 3 guesses, it prompts the player to teach it the identity, appending the character and trait vector directly to the database.

---

## 🌐 Live Demo

🎯 **Live Web App:** [Try the Character Guesser Here](https://your-vercel-link.vercel.app)

The interactive web interface features:
- Dynamic question routing based on real-time candidate entropy.
- Live confidence tracking meter updating after every response.
- Automated fallback to a community teaching interface after 3 incorrect deduction attempts.

---

## 🧮 Dataset & Trait Architecture

- **Characters:** 200+ curated Marvel and DC heroes, anti-heroes, and villains.
- **Trait Dimensions:** 30 binary and fuzzy feature columns covering:
  - **Universes & Alignments:** Marvel, DC, Hero/Good, Villain/Bad.
  - **Species & Biology:** Human, Mutant, Alien/Deity, Cyborg.
  - **Physical Characteristics:** Hair color, eye color, height/build, baldness, non-human skin.
  - **Powers & Archetypes:** Flight, super strength, healing factor, energy blasts, magic/mystic, tech gadgets, martial arts mastery.

---

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


📋 Technical Implementation1. Bayesian Probability UpdateRather than binary pruning, each character maintains a prior probability $P(C_i)$. When an answer with user weight $w$ is submitted, the likelihood is calculated across trait values $T_{ij}$:$$L(C_i) = (T_{ij} \cdot w) + ((1.0 - T_{ij}) \cdot (1.0 - w))$$Probabilities are updated and normalized:$$P(C_i \mid \text{answer}) = \frac{P(C_i) \cdot L(C_i)}{\sum_k P(C_k) \cdot L(C_k)}$$2. Shannon Entropy Question SelectionThe engine evaluates all unasked questions to select the one that yields the maximum Expected Information Gain:$$IG(Q) = H(\text{Current}) - \Big(P(\text{Yes})H(\text{Yes}) + P(\text{No})H(\text{No})\Big)$$3. Dynamic Database LearningWhen 3 guesses fail:The engine yields to the player and opens a submission prompt.The player names the character.The server writes the character to Neon PostgreSQL and maps the answers given during the session directly into the character_trait relationship table.🛠️ Tools & TechnologiesLanguage: Python 3.11Backend Framework: FastAPI, UvicornMathematical Computation: NumPyDatabase & ORM: PostgreSQL (Neon Serverless), SQLAlchemyDeployment Platform: Vercel (Serverless Functions)Frontend: Responsive Vanilla HTML5, CSS3, JavaScript (Fetch API)⚙️ How to Run Locally1. Clone this repositoryBashgit clone [https://github.com/MaheliN1218/CharacterGuessor.git](https://github.com/MaheliN1218/CharacterGuessor.git)
cd CharacterGuessor
2. Create and activate a virtual environmentBashpython -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
3. Install dependenciesBashpip install -r requirements.txt
4. Configure Database and SeedSet your PostgreSQL connection string:Bash# Windows PowerShell
$env:DATABASE_URL="your-postgresql-connection-string"

# Run migrations
python mig.py
5. Launch ApplicationBashuvicorn main:app --reload
Open http://localhost:8000 in your browser.
