# Data Science, Machine Learning et Deep Learning

Parcours IAS, Centrale Lille. Les termes techniques restent en anglais, parce que
c'est en anglais que vous les retrouverez partout ailleurs.

## Ce que contient ce dossier aujourd'hui

### Jour 1 : théorie, de S0 à S4 et le non supervisé

- `diapositives/J1_theorie_s0_s4.pdf` : le diaporama projeté

### Jour 2 : travaux dirigés sur les mêmes sujets

- `notebooks/TD1_prise_en_main.ipynb`
- `notebooks/TD2_cadrer_nettoyer.ipynb`
- `notebooks/TD3_supervise.ipynb`
- `notebooks/TD4_reseaux.ipynb`
- `notebooks/TD5_temporel.ipynb`
- `notebooks/TD6_non_supervise.ipynb`

### Jour 3 : texte, LLM et RAG

- `diapositives/J3_theorie_s5_s6.pdf` : le diaporama projeté
- `notebooks/TD7_texte.ipynb`
- `notebooks/TD8_rag.ipynb`

## Mise en route

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate      # macOS et Linux
pip install -r requirements.txt
python verifier_env.py
```

`verifier_env.py` vérifie votre version de Python, vos bibliothèques et vos
fichiers de données, et vous dit exactement ce qui manque.

## À faire AVANT la séance, chez vous

Le jour 3 utilise deux modèles pré-entraînés. **Téléchargez-les chez vous**, pas
en salle : trente téléchargements simultanés satureront le réseau et vous
perdrez l'après-midi.

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (environ 470 Mo)
- `Qwen/Qwen2.5-1.5B-Instruct` (environ 3,1 Go)

```bash
python telecharger_modeles.py
```

## Comment travailler

En **binôme**, et vous changez de clavier à chaque étape. Celui qui ne tape pas lit
l'énoncé à voix haute et vérifie.

**Vous n'avez pas besoin de lever la main pour savoir si vous avez juste.** Chaque
exercice se termine par une cellule de vérification qui vous le dit, et qui explique
ce qui ne va pas si ce n'est pas bon. Les indices sont repliés sous chaque exercice :
ouvrez-les plutôt que d'attendre.

## Les données

Toutes en open data, et leurs licences sont dans `donnees/LICENCES.md`.

