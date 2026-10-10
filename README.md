# Poker Assistant

Surveille ta table de NLHE à l'écran et, **uniquement quand c'est ton tour**, envoie une capture à Claude, puis affiche l'action conseillée dans une petite fenêtre toujours au premier plan (et dans la console).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...                   # Windows : set ANTHROPIC_API_KEY=...
```

Python 3.11+. Sous Linux, tkinter s'installe à part (`sudo apt install python3-tk`).
Sous macOS, le raccourci clavier global demande l'autorisation « Accessibilité » pour le terminal.

## 1. Calibration

Ouvre ta table, de préférence **quand c'est ton tour** (boutons rouges visibles), puis :

```bash
python calibrate.py              # capture l'écran 1 au bout de 3 s ; --monitor 2 pour un autre écran
```

1. Trace le rectangle de **la table**, puis Entrée.
2. Trace le rectangle de **la barre d'actions** (zone des boutons FOLD / CALL / RAISE), puis Entrée.
3. Indique si c'est ton tour : la barre est enregistrée comme template de référence
   (`templates/turn_bar.png` si c'est ton tour, `templates/wait_bar.png` en présélection).

Pour ajouter le second template sans refaire les régions, relance pendant l'autre état :
`python calibrate.py --template-only`. Tu peux aussi calibrer depuis une capture plein écran
enregistrée : `python calibrate.py --image capture.png`.

## 2. Vérifier la détection

Dépose des captures dans `samples/` en indiquant l'état dans le nom
(`montour_01.png`, `attente_01.png`, …), plein écran ou table seule, puis :

```bash
python test_detection.py --debug     # images annotées dans samples/_debug/
```

Le script affiche, pour chaque image, la proportion de rouge, le nombre de boutons détectés et
les scores de template, et signale les erreurs. Tu peux aussi tester en direct sans payer l'API :
`python main.py --dry-run`.

## 3. Utilisation

```bash
python main.py            # fenêtre + console
python main.py --no-ui    # console seule
python main.py --image table.png   # analyse ponctuelle d'une capture
```

- **Pause / reprise** : `Ctrl+Alt+P` (configurable : `ui.hotkey`) ou le bouton de la fenêtre.
- **Journal** : chaque décision est enregistrée dans `logs/AAAA-MM-JJ/` : la capture (`.png`),
  la réponse (`.txt`) et une ligne dans `decisions.jsonl`.

## Fonctionnement

| Module | Rôle |
|---|---|
| `poker_assistant/capture.py` | capture de la région avec `mss` |
| `poker_assistant/detection.py` | détection locale du tour (sans API) |
| `poker_assistant/state.py` | anti-doublon : un appel par tour |
| `poker_assistant/claude_client.py` | appel Claude (image en base64), gestion des erreurs |
| `poker_assistant/engine.py` | boucle capture → détection → appel en arrière-plan |
| `poker_assistant/ui.py`, `hotkeys.py`, `journal.py` | fenêtre, raccourci clavier, journal |

**Détection du tour** : dans la barre d'actions, l'outil cherche les gros boutons rouges
(masque HSV + composantes connexes : au moins `min_buttons` boutons et `red_ratio_min` de pixels
rouges). La barre « Présélection de la prochaine action » n'en contient pas. En mode `hybrid`
(défaut), les templates calibrés servent de garde-fou : la barre doit ressembler davantage au
template « tour » qu'au template « attente ». Un changement d'état n'est validé qu'après
`stable_frames` captures identiques.

**Anti-doublon** : l'API n'est appelée qu'au passage « pas mon tour → mon tour », et seulement si
la table (barre d'actions masquée, donc sans le compte à rebours) a changé depuis le dernier appel.
Un clignotement de la barre ne déclenche donc rien. Avec `dedupe.recall_on_change: true`, un nouvel
appel a aussi lieu si la table change nettement pendant ton tour.

**Erreurs API** : timeout, limite de requêtes (429), surcharge et coupures réseau sont réessayés
par le SDK (`claude.max_retries`), puis par l'outil jusqu'à `dedupe.max_retries_per_turn` fois,
tant que c'est encore ton tour. La boucle continue quoi qu'il arrive.

## Réglages utiles (`config.yaml`)

- `claude.model` / `claude.effort` : modèle (défaut `claude-opus-5-5`) et effort (`low` = le plus rapide).
  Pour aller encore plus vite ou réduire le coût : `claude-sonnet-5-5` ou `claude-haiku-5-5`.
- Faux négatifs (tour non détecté) : baisse `detection.red_ratio_min` ou `min_buttons`, ou
  élargis `red_hue_low` / baisse `red_sat_min` si le rouge de ton client est terne.
- Faux positifs : recalibre les templates, ou passe `detection.method` à `template`.

`calibrate.py` réécrit `config.yaml` (les commentaires du fichier sont alors perdus).

Lancer les tests : `pip install pytest && python -m pytest tests`.

> Vérifie que l'usage d'un outil d'aide en temps réel est autorisé par les conditions de ta salle.
