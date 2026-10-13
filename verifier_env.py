"""Verifie que votre environnement est pret. A lancer avant la premiere seance.

    python verifier_env.py
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent
MINI = (3, 10)

PAQUETS = ["numpy", "pandas", "matplotlib", "sklearn", "pyarrow"]
PAQUETS_J3 = ["torch", "transformers", "sentence_transformers"]


def main():
    souci = 0

    print(f"Python {sys.version_info.major}.{sys.version_info.minor}", end="  ")
    if sys.version_info < MINI:
        print(f"TROP ANCIEN, il faut au moins {MINI[0]}.{MINI[1]}")
        souci += 1
    else:
        print("ok")

    for nom in PAQUETS:
        try:
            __import__(nom)
            print(f"{nom:<22} ok")
        except ImportError:
            print(f"{nom:<22} MANQUE   ->  pip install -r requirements.txt")
            souci += 1

    for nom in PAQUETS_J3:
        try:
            __import__(nom)
            print(f"{nom:<22} ok  (jour 3)")
        except ImportError:
            print(f"{nom:<22} absent, necessaire seulement au jour 3")

    donnees = RACINE / "donnees"
    if donnees.is_dir():
        fichiers = sorted(p.name for p in donnees.iterdir() if p.suffix in (".parquet", ".csv"))
        print(f"\ndonnees : {len(fichiers)} fichier(s) -> {fichiers}")
    else:
        print("\ndonnees : dossier absent (normal avant le jour 2)")

    notebooks = RACINE / "notebooks"
    if notebooks.is_dir():
        n = len(list(notebooks.glob("*.ipynb")))
        print(f"notebooks : {n}")

    print()
    if souci:
        print(f"{souci} probleme(s). Corrigez-les AVANT la seance.")
        return 1
    print("Tout est pret.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
