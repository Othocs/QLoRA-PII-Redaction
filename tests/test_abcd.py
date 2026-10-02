"""ABCD loader: labels derived from the scenario card, consistency filter, document format."""

from prepare_abcd import build, label_text

SCENARIO = {
    "personal": {"customer_name": "crystal minh", "email": "cminh730@email.com",
                 "phone": "(977) 625-2661", "username": "cminh730"},
    "order": {"street_address": "6821 1st ave", "city": "san mateo", "zip_code": "75227",
              "order_id": "3348917502"},
}  # fmt: skip


def labels(text, speaker="customer"):
    spans, _ = label_text(text, SCENARIO, speaker)
    return [(s.label, text[s.start : s.end]) for s in spans]


def test_names_parts_and_case():
    assert labels("Crystal Minh here") == [("GIVENNAME", "Crystal"), ("SURNAME", "Minh")]
    assert labels("thanks crystal!") == [("GIVENNAME", "crystal")]


def test_phone_variants_and_ignore_and_kept():
    assert labels("call (977) 625-2661") == [("TELEPHONENUM", "(977) 625-2661")]
    assert labels("it's 977-625-2661") == [("TELEPHONENUM", "977-625-2661")]
    assert labels("Username: cminh730, email cminh730@email.com, order 3348917502") == [
        ("IGNORE", "cminh730"), ("EMAIL", "cminh730@email.com")]  # fmt: skip


def test_address_new_house_number_and_agent_intro():
    assert labels("6821 1st Ave, San Mateo 75227") == [
        ("BUILDINGNUM", "6821"), ("STREET", "1st Ave"), ("CITY", "San Mateo"),
        ("ZIPCODE", "75227")]  # fmt: skip
    assert labels("new address: 9090 1st ave") == [("BUILDINGNUM", "9090"), ("STREET", "1st ave")]
    assert labels("Hi, this is Amy from support", "agent") == [("GIVENNAME", "Amy")]
    assert labels("Hi, this is Amy from support", "customer") == []


def test_build_drops_unmatched_delex_and_formats():
    convo = {"convo_id": 7, "scenario": SCENARIO,
             "original": [["agent", "Name please?"], ["customer", "Crystal Minh"],
                          ["action", "Account has been pulled up for Crystal Minh."]],
             "delexed": [{"text": "name please?"}, {"text": "crystal minh"},
                         {"text": "account has been pulled up for crystal minh."}]}  # fmt: skip
    ex, why = build(convo)
    assert why == "" and ex.id == "abcd-7"
    assert ex.text == "Agent: Name please?\nCustomer: Crystal Minh"  # action turn dropped
    assert [(s.label, s.text) for s in ex.spans] == [("GIVENNAME", "Crystal"), ("SURNAME", "Minh")]
    convo["original"][1][1] = "my email is cminh730 at email dot com"
    convo["delexed"][1]["text"] = "my email is <email>"
    assert build(convo) == (None, "unmatched <email>")
