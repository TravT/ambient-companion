# Evaluation harness (PRJ-12 Phase D)

Measures what the companion gets right and how long it takes, per question kind, backend and image size, so
routing and image-size defaults are decided by data and not by one or two frames.

```bash
# 1. a synthetic set with known answers (validates the harness; baseline before real photos exist)
python3 -m eval.make_synthetic data/eval/synthetic

# 2. run it against every Tier 2 backend that is awake right now (nothing is ever woken or switched)
python3 -m eval.run_eval --cases data/eval/synthetic/cases.yaml --backends dell,satellite,gpu --px 512,768,1024
```

Results go to a JSONL file next to the cases file (one row per question: answer, score, latency, tokens, click
error) and a markdown summary is printed: accuracy and median latency per (kind, backend, size), once over all
questions and once over the first question of each image only (a cold image; later questions on the same image
are fast because the server caches the image).

`data/eval/` is git-ignored: photos and answers stay on the machine.

## Real photos

Put photos in `data/eval/photos/` and describe them in `data/eval/photos/cases.yaml`:

```yaml
- id: desk-01
  image: desk-01.jpg                 # relative to this file
  questions:
    - {q: "Is there a laptop on the desk?", kind: presence, expect: "yes"}
    - {q: "How many cups are there?", kind: count, expect: 3}
    - {q: "What does the label say?", kind: read, expect_text: ["500 mg", "ibuprofeno"]}
    - {q: "Where is the remote?", kind: locate, expect_click: [1900, 1300], tolerance: 300}   # pixels of the ORIGINAL photo
    - {q: "Describe the scene.", kind: describe, all_of: ["desk"], any_of: ["lamp", "laptop"]}
```

Kinds and scoring: `presence` yes/no (English and Portuguese answers), `count` (digits or number words),
`read` (each expected phrase found, tolerating about 20% character error), `locate` (click target within
`tolerance` pixels of the true centre), `describe` (keyword rubric). Failures are listed in the JSONL for review.

### What to bring (about 40 photos; phone shots are best; any year)

| Group | Count | Subjects | Tests |
| :--- | ---: | :--- | :--- |
| Rooms and desks | 8 | A desk, shelf or counter with 5+ objects; some with one object removed | yes/no, describe |
| Absence | 5 | Scenes where the thing asked about is missing | false "Yes" answers |
| Text to read | 10 | Labels, signs, book spines, packages, screens; half Portuguese; some small print | reading, 768 vs 1024 px |
| Counting and attributes | 5 | Several cups, plants or books; one clear color or size difference | counting |
| Location | 6 | A clear object at a known place; plus 3 screenshots | boxes, click targets |
| Bad conditions | 6 | Dim light, blur, glare, far away, odd angle | robustness |

Plus 3-4 fresh frames from the S20 camera, since that is what the live path sees. Please avoid photos with
faces, documents, IDs or financial information. For each photo one line of what is true (for example "3 cups,
one red; label says dosage 500 mg") is enough; the questions can be drafted from it. If the photos are viewed by
an assistant that runs in the cloud, they leave the machine; the harness itself only talks to your own servers.

## Findings so far (synthetic set, satellite, 2026-10-09)

* 49 of 54 questions fully correct. Locating: click error 0-6 px at 512, 768 and 1024 px. Counting, colors and presence: correct (the one "wrong" yes/no was a badly drawn plate in the synthetic image).
* Reading (6 text images, English and Portuguese, 14/24/48 px fonts): 94% at 512 px, 89% at 768 px, 100% at 1024 px. At 768 px a Portuguese request was half translated into English; at 512 px small print was misread ("Dipiramina" for "Dipirona").
* **Prompt wording did not help**: plain, English "transcribe exactly, do not translate", Portuguese equivalent and "OCR:" phrasings all gave the same output. Image size is the lever. 1024 px costs about 1.8x the time of 768 px (satellite 51 s against 29 s cold), so the reading budget (`AMBIENT_READ_PX`, 768 by default) is a trade-off to settle with real photos.
