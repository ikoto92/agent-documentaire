# Publier le projet sur GitHub et le mettre en ligne

## 1. Vérifier qu'aucun secret ne part sur GitHub

Le fichier `.env` contient ta clé API : il est listé dans `.gitignore`, il ne sera donc **pas** envoyé.
Vérifie-le avant chaque premier envoi :

```powershell
git status          # .env ne doit PAS apparaître dans la liste
```

Si ta clé a déjà été montrée à quelqu'un (capture d'écran, message), supprime-la et crée-en une
nouvelle sur https://aistudio.google.com/apikey, puis mets la nouvelle dans `.env`.

## 2. Mettre le projet sur GitHub

1. Sur https://github.com/new, crée un dépôt **public** nommé `agent-documentaire`,
   **sans** README ni .gitignore (le projet les a déjà).
2. Dans PowerShell, depuis le dossier du projet :

```powershell
cd D:\COURS\agent-documentaire\agent-documentaire
git init
git add .
git status                      # vérifier : pas de .env, pas de .venv
git commit -m "Agent documentaire IA : RAG, tool calling, évaluation, interface web"
git branch -M main
git remote add origin https://github.com/ikoto92/agent-documentaire.git
git push -u origin main
```

Si Git demande de s'identifier, connecte-toi avec ton compte GitHub dans la fenêtre qui s'ouvre.

3. Sur la page du dépôt, onglet **Actions** : la CI lance les tests automatiquement.
   Une coche verte = tous les tests passent. C'est une preuve visible pour un recruteur.

## 3. Mettre l'interface en ligne (Streamlit Community Cloud, gratuit)

1. Va sur https://share.streamlit.io et connecte-toi avec ton compte GitHub.
2. **Create app** → choisis le dépôt `ikoto92/agent-documentaire`, la branche `main`
   et le fichier `app_web.py`.
3. Dans **Advanced settings → Secrets**, colle le contenu de `.streamlit/secrets.toml.example`
   en remplaçant `colle-ta-cle-ici` par ta vraie clé.
4. **Deploy**. Après quelques minutes, tu obtiens une adresse publique du type
   `https://agent-documentaire.streamlit.app`.

Ajoute ensuite cette adresse dans le README, sur ton CV, ton portfolio et ton profil LinkedIn.

Les conditions des offres gratuites changent régulièrement : vérifie-les sur le site de l'hébergeur.
Une application Streamlit gratuite peut se mettre en veille si personne ne l'utilise ;
le premier visiteur la réveille en une minute environ.

## 4. Après une modification

```powershell
git add .
git commit -m "Description courte de la modification"
git push
```

L'application en ligne se met à jour toute seule après chaque `git push`.
