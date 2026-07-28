# Build the report. `make figures` regenerates the plots from the run summaries
# first, so the PDF can never be built against stale data.
MAIN = main

$(MAIN).pdf: $(MAIN).tex references.bib
	pdflatex -interaction=nonstopmode $(MAIN)
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)

# Every figure, including the two Background schematics, is matplotlib: the
# glob has to catch arch_diagrams.py as well as fig*.py, or the schematics go
# stale while the data figures update.
figures:
	cd scripts && for f in arch_diagrams.py fig*.py; do python "$$f"; done
	mkdir -p figures && cp ../results/figures/*.pdf figures/

clean:
	rm -f $(MAIN).pdf *.aux *.bbl *.blg *.log *.out

.PHONY: figures clean
