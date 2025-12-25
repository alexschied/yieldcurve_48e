include .env
export

ENV=ec48e-recession

.PHONY: env run tables report clean

env:
	conda env create -f environment.yml

run:
	conda run -n $(ENV) python -m experiments.run_full_pipeline


tables:
	conda run -n $(ENV) python experiments/export_tables.py

report:
	cd reports && pdflatex main.tex

clean:
	rm -rf outputs reports/figures/*.pdf reports/tables/*.tex
