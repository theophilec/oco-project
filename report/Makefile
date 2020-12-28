# Depends: rubber
PAPER=main.tex

# Build (xelatex not compatible with rubber, need to fix later.)
all:
	xelatex $(PAPER)
	bibtex main
	xelatex $(PAPER)
	xelatex $(PAPER)

clean:
	rubber --clean $(PAPER)