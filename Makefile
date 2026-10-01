# PII redaction gateway. `make help` lists targets.
UV ?= uv
EVAL_LIMIT ?= 200
SYSTEMS ?= presidio,gliner_knowledgator,gliner_nvidia
TESTSETS ?= dev
EXTRAS ?= --extra data --extra presidio --extra presidio-lg --extra gliner --extra serve

.PHONY: help setup data eval-data audit audit-score eval summary support-desk test test-all lint format train serve demo

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-13s %s\n", $$1, $$2}'

setup: ## install dependencies (uv)
	$(UV) sync $(EXTRAS)

data: ## download OpenPII, keep English, build splits + MinHash dedup
	$(UV) run python data/prepare_openpii.py all

eval-data: ## build the out-of-distribution test sets (Nemotron, TAB, Gretel, OpenPII non-EN)
	$(UV) run python data/prepare_eval_sets.py

audit: ## sample 200 training docs into review sheets (data/audit/)
	$(UV) run python data/audit/sample_audit.py

audit-score: ## score the filled audit sheets into data/audit/AUDIT.md
	$(UV) run python data/audit/score_audit.py

eval: ## run baselines (SYSTEMS) on TESTSETS, first EVAL_LIMIT examples
	$(UV) run python -m eval.run_eval --systems $(SYSTEMS) --testsets $(TESTSETS) $(if $(EVAL_LIMIT),--limit $(EVAL_LIMIT))

summary: ## rebuild results/SUMMARY.md and the README table
	$(UV) run python -m eval.summarize

support-desk: ## fill offsets and validate the support-desk set
	$(UV) run python data/support_desk/check_support_desk.py --fill

test: ## fast tests (no model downloads)
	$(UV) run pytest -q -m "not slow"

test-all: ## all tests, including ones that load spaCy / Presidio
	$(UV) run pytest -q

lint: ## ruff check + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

train: ## week 2
	@echo "not implemented yet (week 2): training/train_lora.py"; exit 1
serve: ## week 4
	@echo "not implemented yet (week 4): src/pii_gateway/api.py"; exit 1
demo: ## week 5
	@echo "not implemented yet (week 5): demo/app.py"; exit 1
