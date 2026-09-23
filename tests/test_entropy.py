"""Thin entropy tests (asserts ported from examples/entropy_split_demo)."""

from __future__ import annotations

from jevtree.core.entropy import (
    best_split,
    entropy,
    gain_ratio,
    information_gain,
    split_info,
)

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


def test_parent_entropy_and_outlook_best_ig() -> None:
    labels = [r[LABEL] for r in PLAY_TENNIS]
    parent_h = entropy(labels)
    assert abs(parent_h - 0.940) < 0.01

    ig_by_feat = {
        fk: information_gain(labels, [r[fk] for r in PLAY_TENNIS]) for fk in FEATURES
    }
    assert max(ig_by_feat, key=ig_by_feat.get) == "Outlook"  # type: ignore[arg-type]
    assert abs(ig_by_feat["Outlook"] - 0.246) < 0.01

    split_gain = best_split(PLAY_TENNIS, LABEL, FEATURES, criterion="gain")
    assert split_gain["feature"] == "Outlook"
    assert abs(parent_h - split_gain["parent_entropy"]) < 1e-12


def test_edge_cases() -> None:
    assert entropy([]) == 0.0
    assert entropy(["Yes", "Yes", "Yes"]) == 0.0
    assert information_gain(["a", "b"], ["x", "x"]) == 0.0
    assert gain_ratio(["a", "b"], ["x", "x"]) == 0.0
    assert split_info(["x", "x"]) == 0.0
    empty = best_split([], LABEL, FEATURES)
    assert empty["feature"] is None and empty["score"] == 0.0
    no_feat = best_split(PLAY_TENNIS, LABEL, [])
    assert no_feat["feature"] is None and no_feat["score"] == 0.0


if __name__ == "__main__":
    test_parent_entropy_and_outlook_best_ig()
    test_edge_cases()
    print("ok")
