#!/usr/bin/env python3
"""Demo: information entropy / IG / gain_ratio on classic Play Tennis table.

Asserts Outlook has the highest information gain (textbook result).
Run from Meta-Jev root:
  PYTHONPATH=src python3 examples/entropy_split_demo.py
"""

from __future__ import annotations

from meta_jev.core.entropy import (
    best_split,
    entropy,
    gain_ratio,
    information_gain,
    split_info,
)

# Classic Play Tennis (Quinlan) - 14 rows
PLAY_TENNIS: list[dict] = [
    {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "Play": "No"},
    {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Strong", "Play": "No"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play": "No"},
    {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "Play": "No"},
    {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "Play": "No"},
]

FEATURES = ["Outlook", "Temperature", "Humidity", "Wind"]
LABEL = "Play"


def main() -> None:
    labels = [r[LABEL] for r in PLAY_TENNIS]
    parent_h = entropy(labels)
    print("Play Tennis rows:", len(PLAY_TENNIS))
    print("Parent entropy H(Play): {:.6f}".format(parent_h))
    print()

    print("{:<12} {:>10} {:>12} {:>12}".format("Feature", "IG", "SplitInfo", "GainRatio"))
    print("-" * 50)
    ig_by_feat: dict[str, float] = {}
    for fk in FEATURES:
        fvals = [r[fk] for r in PLAY_TENNIS]
        ig = information_gain(labels, fvals)
        si = split_info(fvals)
        gr = gain_ratio(labels, fvals)
        ig_by_feat[fk] = ig
        print("{:<12} {:10.6f} {:12.6f} {:12.6f}".format(fk, ig, si, gr))

    print()
    split_gain = best_split(PLAY_TENNIS, LABEL, FEATURES, criterion="gain")
    split_gr = best_split(PLAY_TENNIS, LABEL, FEATURES, criterion="gain_ratio")
    print("Best split (gain):       feature={!r}  score={:.6f}".format(
        split_gain["feature"], split_gain["score"]))
    print("Best split (gain_ratio): feature={!r}  score={:.6f}".format(
        split_gr["feature"], split_gr["score"]))
    print("Parent entropy (via best_split): {:.6f}".format(split_gain["parent_entropy"]))

    # --- asserts (no pytest required) ---
    assert abs(parent_h - 0.940) < 0.01, "expected H~=0.940, got {}".format(parent_h)
    assert abs(parent_h - split_gain["parent_entropy"]) < 1e-12

    # Outlook has highest IG (classic textbook)
    best_ig_feat = max(ig_by_feat, key=ig_by_feat.get)  # type: ignore[arg-type]
    assert best_ig_feat == "Outlook", "expected Outlook highest IG, got {}: {}".format(
        best_ig_feat, ig_by_feat)
    assert split_gain["feature"] == "Outlook"
    assert abs(ig_by_feat["Outlook"] - 0.246) < 0.01

    # edge cases
    assert entropy([]) == 0.0
    assert entropy(["Yes", "Yes", "Yes"]) == 0.0
    assert information_gain(["a", "b"], ["x", "x"]) == 0.0
    assert gain_ratio(["a", "b"], ["x", "x"]) == 0.0
    empty = best_split([], LABEL, FEATURES)
    assert empty["feature"] is None and empty["score"] == 0.0
    no_feat = best_split(PLAY_TENNIS, LABEL, [])
    assert no_feat["feature"] is None and no_feat["score"] == 0.0

    print()
    print("All asserts passed.")


if __name__ == "__main__":
    main()
