# ABCD test set: real, human-typed support chats

Built by `data/prepare_abcd.py` into `data/processed/abcd.jsonl` (gitignored, rebuilt on pods).

**Source.** ASAPP's [Action-Based Conversations Dataset](https://github.com/asappresearch/abcd) (MIT licence), test split: 1,004 conversations. Trained call-centre agents chat with crowd workers, who play a customer from a scenario card holding fictional details.

- **The text is typed by people.** That is the point of this test: it is real support language, unlike our LLM-drafted support-desk sets.
- **The PII values are fictional,** copied from the card.

## Labels

Labels come from matching the card's values in the text: whole word, case-insensitive, every occurrence.

| Card value | Label |
| --- | --- |
| customer_name | GIVENNAME / SURNAME |
| email | EMAIL |
| phone | TELEPHONENUM (also digit variants) |
| street address | BUILDINGNUM + STREET |
| city | CITY |
| zip | ZIPCODE |
| username, account ID, PIN, password, security answer | IGNORE (outside our label set) |

A few cases not on the card are also labelled:

- a house number right before a labelled street, when the customer types a new address;
- an agent introducing themselves by name ("this is Amy from…"), as GIVENNAME;
- a username typed after "username is / :", as IGNORE.

Order IDs, the state, membership level, products and amounts are not personal and stay unlabelled.

**Consistency filter.** ABCD's delexicalised copy marks email, phone, zip, street, username, account ID and PIN with tokens. A conversation is dropped when those tokens show a value our matching missed, which would otherwise count as a false leak. This dropped 2 of 1,004 conversations; 1,002 are kept, 787 of them with PII.

| Label | Spans |
| --- | ---: |
| GIVENNAME | 1,097 |
| SURNAME | 866 |
| EMAIL | 230 |
| ZIPCODE | 172 |
| CITY | 126 |
| STREET | 115 |
| BUILDINGNUM | 114 |
| TELEPHONENUM | 71 |
| IGNORE | 617 |

## Limitations

- **Few names and cities.** ABCD reuses only 10 customer names and 9 cities across the whole test split, the same pool as its training split. Name and city results mostly show how a model handles those 10 names across many conversational contexts.
- **More varied values.** Emails, phones, zips and street addresses vary by conversation.
- **Misspelled names.** Names are not delexicalised, so a misspelled name would go unlabelled. The spot check (`spotcheck_40.csv`) found none.
