PY ?= .venv/bin/python

.PHONY: setup fetch pipeline notebooks app demo test all clean-results

setup:            ## create the virtual environment and install requirements
	uv venv --python 3.13 .venv || python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

fetch:            ## download every open source and re-parse data/raw
	$(PY) pipeline/s00_fetch_data.py

pipeline:         ## stages 1-11: datasets ... hardenability, hardening + tempering, assumptions, kinetic uncertainty
	$(PY) pipeline/run_all.py

notebooks:        ## rebuild and execute the presentation notebooks from results/
	$(PY) notebooks/make_notebooks.py

app:              ## launch the interactive furnace-vs-air what-if tool
	$(PY) app/app.py

demo:             ## launch the clean one-page live demo (opens http://127.0.0.1:8050)
	$(PY) app/demo/server.py

test:             ## physics, data and model tests
	$(PY) -m pytest tests -q

report:           ## REPORT.md, slide deck and manuscript from results/
	.venv/bin/python reports/build_report.py
	.venv/bin/python reports/deck/build_deck.py
	.venv/bin/python reports/paper/build_paper.py

all: pipeline notebooks report test
