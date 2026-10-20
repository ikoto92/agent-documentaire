"""Affiche les modèles disponibles chez votre fournisseur, avec votre clé.

Usage : python list_models.py
Utile pour choisir CHAT_MODEL et CHAT_MODEL_FALLBACK dans .env.
"""

from app.config import get_settings
from app.factory import make_client


def main() -> None:
    settings = get_settings()
    models = sorted(m.id.removeprefix("models/") for m in make_client(settings).models.list())
    print(f"{len(models)} modèles disponibles chez {settings.base_url} :\n")
    for name in models:
        marks = []
        if name == settings.chat_model:
            marks.append("CHAT_MODEL")
        if name in settings.fallback_models:
            marks.append("CHAT_MODEL_FALLBACK")
        if name == settings.embed_model:
            marks.append("EMBED_MODEL")
        print(f"  {name}" + (f"   <- {', '.join(marks)}" if marks else ""))


if __name__ == "__main__":
    main()
