"""Make an app voice profile from a fit: profile.py <best.json> <id> "<Name>" ["note"]

Writes src/voices/local/<id>.json (gitignored), which the app lists under Voice.
"""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import fit  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    best, pid, name = sys.argv[1:4]
    if not re.fullmatch(r"[\w-]+", pid):
        sys.exit(f"id {pid!r}: letters, digits, - and _ only")
    p = json.load(open(best))["params"]
    out = {"name": name, **({"note": sys.argv[4]} if len(sys.argv) > 4 else {}), "fit": str(pathlib.Path(best).resolve()), **fit.profile(p)}
    path = REPO / "src" / "voices" / "local" / f"{pid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main()
