"""requirements.txt, which the course installs from, lists exactly the packages
pyproject.toml declares as dependencies and extras, so the two cannot drift.

pyproject.toml is read with regular expressions rather than tomllib, which only
exists from Python 3.11, and the check also runs on 3.10."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _norm(spec):
    return re.split(r"[<>=!~\[;]", spec.strip())[0].strip().lower()


def _declared(text):
    deps = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.S | re.M).group(1)
    extras = re.search(r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[)", text, re.S | re.M).group(1)
    return {_norm(m) for m in re.findall(r'"([^"]+)"', deps + extras)}


def test_requirements_match_pyproject():
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as fh:
        declared = _declared(fh.read())
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as fh:
        listed = {_norm(l) for l in fh if l.strip() and not l.startswith("#")}
    assert declared, "no packages parsed from pyproject.toml"
    assert listed == declared, (sorted(listed), sorted(declared))
