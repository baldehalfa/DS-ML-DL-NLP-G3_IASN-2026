""" Telecharge les modeles pre-entraines du jour 3. A lancer CHEZ VOUS.

    python telecharger_modeles.py
"""
MODELES = [
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "environ 470 Mo"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "environ 3,1 Go"),
]


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub manque. Lancez d'abord : pip install -r requirements.txt")
        return 1
    for nom, taille in MODELES:
        print(f"\n{nom}  ({taille})")
        chemin = snapshot_download(
            nom, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
        print(f"  -> {chemin}")
    print("\nTermine. Les modeles sont en cache, la seance n'aura rien a telecharger.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
