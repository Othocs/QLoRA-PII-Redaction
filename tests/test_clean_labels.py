"""Label-noise rules: positives come from audited OpenPII rows, negatives from the
false positives found while validating the rules on unseen training data."""

import pytest
from clean_labels import RULES, flags

from pii_gateway.spans import Example, Span


def ex_with(text: str, value: str, label: str) -> Example:
    i = text.index(value)
    return Example("t", text, [Span(i, i + len(value), label, value)])


NOISE = [
    ("total funds collected: $5212500387010542; average", "5212500387010542", "CREDITCARDNUMBER"),
    ("the monthly rent of £6581559950053269 is payable", "6581559950053269", "CREDITCARDNUMBER"),
    ("pollination efficiency increased by 81 % across sites", "81", "AGE"),
    ("reflects on how 64 years of experience reshaped", "64", "AGE"),
    ("for participants ranging from 38 to 15 years old", "15", "AGE"),
    ("records of individuals aged 74 to 53 were processed", "74", "AGE"),
    ("and a gender ratio (F) for each functional area", "F", "SEX"),
    ("regardless of Female or Male background", "Female", "SEX"),
    ("ensure proper 68702.63793 compliance", "68702.63793", "TAXNUM"),
]
PERSONAL = [
    ("a service fee payable via 2204290235773370.", "2204290235773370", "CREDITCARDNUMBER"),
    ("the card ending in 4111111111111111 will be charged", "4111111111111111", "CREDITCARDNUMBER"),
    ("The lead researcher, a Non-binary scholar aged 80, reported", "80", "AGE"),
    ("your current 49-year-old profile", "49", "AGE"),
    ("you have indicated a gender preference of Genderqueer.", "Genderqueer", "GENDER"),
    ("Age: 39, Sex: Other, Gender Identity: Agender.", "39", "AGE"),
    ("Tax ID No.: 27157.03506", "27157.03506", "TAXNUM"),
]


@pytest.mark.parametrize("text,value,label", NOISE)
def test_flags_noise(text, value, label):
    assert flags(ex_with(text, value, label)), text


@pytest.mark.parametrize("text,value,label", PERSONAL)
def test_keeps_personal_data(text, value, label):
    assert not flags(ex_with(text, value, label)), text


def test_rules_ignore_other_labels():
    ex = ex_with("increased by 81 % overall", "81", "BUILDINGNUM")
    assert not any(rule(ex.text, ex.spans[0]) for rule in RULES.values())
