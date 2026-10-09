#!/usr/bin/env python3
"""
Synthetic evaluation set with KNOWN answers (PRJ-12 Phase D).

Draws images with PIL (counts, colors, rendered text at several sizes in English and Portuguese, a login
page with known element positions, absence scenes) and writes cases.yaml. It validates the harness and gives a
baseline before real photos exist; real photos use the same cases.yaml format (see eval/README.md).

  python3 -m eval.make_synthetic data/eval/synthetic
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def counts_scene(path: Path) -> List[Dict[str, Any]]:
    img = Image.new("RGB", (1280, 800), (245, 245, 240))
    d = ImageDraw.Draw(img)
    for i, x in enumerate((200, 480, 760)):
        d.ellipse([x, 150, x + 160, 310], fill=(210, 40, 40))
    for x in (300, 700):
        d.rectangle([x, 480, x + 140, 620], fill=(40, 80, 200))
    img.save(path)
    return [
        {"q": "How many red circles are there?", "kind": "count", "expect": 3},
        {"q": "How many blue squares are there?", "kind": "count", "expect": 2},
        {"q": "Is there a blue square?", "kind": "presence", "expect": "yes"},
        {"q": "Is there a green triangle?", "kind": "presence", "expect": "no"},
    ]


def color_scene(path: Path) -> List[Dict[str, Any]]:
    img = Image.new("RGB", (1280, 800), (230, 230, 230))
    ImageDraw.Draw(img).rectangle([300, 200, 980, 600], fill=(240, 200, 20))
    img.save(path)
    return [{"q": "What color is the large rectangle?", "kind": "describe", "any_of": ["yellow", "amarelo"]}]


def text_scene(path: Path, lines: List[str], size: int) -> None:
    img = Image.new("RGB", (1600, 1200), (252, 252, 248))
    d = ImageDraw.Draw(img)
    y = 120
    for line in lines:
        d.text((90, y), line, fill=(20, 20, 20), font=font(size, bold=size >= 40))
        y += int(size * 2.2)
    img.save(path)


def login_scene(path: Path) -> List[Dict[str, Any]]:
    W, H = 1280, 800
    img = Image.new("RGB", (W, H), (245, 247, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 64], fill=(30, 41, 59))
    d.text((24, 18), "Acme Cloud Console", fill="white", font=font(26, True))
    d.rounded_rectangle([420, 150, 860, 690], 14, fill="white", outline=(203, 213, 225), width=2)
    d.text((470, 180), "Sign in to your account", fill=(15, 23, 42), font=font(28, True))
    d.rectangle([470, 240, 810, 300], fill=(254, 226, 226), outline=(220, 38, 38))
    d.text((484, 260), "Invalid credentials. 2 attempts left.", fill=(153, 27, 27), font=font(16))
    d.text((470, 330), "Username", fill=(51, 65, 85), font=font(16))
    d.rectangle([470, 356, 810, 406], outline=(148, 163, 184), width=2)
    d.text((470, 430), "Password", fill=(51, 65, 85), font=font(16))
    d.rectangle([470, 456, 810, 506], outline=(148, 163, 184), width=2)
    d.rounded_rectangle([470, 540, 810, 596], 8, fill=(37, 99, 235))
    d.text((590, 556), "Sign in", fill="white", font=font(22, True))
    d.text((470, 620), "Forgot password?", fill=(37, 99, 235), font=font(16))
    img.save(path)
    return [
        {"q": "Where is the blue Sign in button?", "kind": "locate", "expect_click": [640, 568], "tolerance": 40},
        {"q": "Where is the Username text input field?", "kind": "locate", "expect_click": [640, 381], "tolerance": 40},
        {"q": "Where is the red error banner?", "kind": "locate", "expect_click": [640, 270], "tolerance": 40},
        {"q": "What does the red banner say?", "kind": "read", "expect_text": ["Invalid credentials", "2 attempts left"]},
    ]


def absence_scene(path: Path) -> List[Dict[str, Any]]:
    img = Image.new("RGB", (1280, 800), (200, 200, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([100, 500, 1180, 700], fill=(120, 90, 60))             # a plain table
    d.ellipse([560, 420, 720, 520], fill=(255, 255, 255))              # a white plate
    img.save(path)
    return [
        {"q": "Is there a cat in the image?", "kind": "presence", "expect": "no"},
        {"q": "Is there a person in the image?", "kind": "presence", "expect": "no"},
        {"q": "Is there a plate on the table?", "kind": "presence", "expect": "yes"},
    ]


def build(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    cases: List[Dict[str, Any]] = []

    def add(cid: str, image: str, questions: List[Dict[str, Any]]) -> None:
        cases.append({"id": cid, "image": image, "questions": questions})

    add("counts", "counts.png", counts_scene(out / "counts.png"))
    add("color", "color.png", color_scene(out / "color.png"))
    add("ui-login", "ui-login.png", login_scene(out / "ui-login.png"))
    add("absence", "absence.png", absence_scene(out / "absence.png"))
    for size in (14, 24, 48):
        text_scene(out / f"text-en-{size}.png", ["IBUPROFEN 500 mg", "Take 1 tablet every 8 hours.", "Expires 03/2028"], size)
        add(f"text-en-{size}", f"text-en-{size}.png", [
            {"q": "Read all the text in the image.", "kind": "read",
             "expect_text": ["IBUPROFEN 500 mg", "Take 1 tablet every 8 hours", "Expires 03/2028"]}])
        text_scene(out / f"text-pt-{size}.png", ["Dipirona sódica 500 mg", "Tome um comprimido a cada 8 horas.", "Validade: 03/2028"], size)
        add(f"text-pt-{size}", f"text-pt-{size}.png", [
            {"q": "Leia todo o texto da imagem.", "kind": "read",
             "expect_text": ["Dipirona sódica 500 mg", "Tome um comprimido a cada 8 horas", "Validade 03/2028"]}])

    try:
        import yaml
        (out / "cases.yaml").write_text(yaml.safe_dump(cases, sort_keys=False, allow_unicode=True))
    except ImportError:                    # JSON is valid YAML
        import json
        (out / "cases.yaml").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    return out / "cases.yaml"


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/eval/synthetic")
    print(build(target))
