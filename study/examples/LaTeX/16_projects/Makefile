# Makefile for LaTeX examples compilation
# Automates building LaTeX documents with proper dependencies

# Compiler and flags
LATEX = pdflatex
BIBTEX = bibtex
BIBER = biber
LATEXMK = latexmk
LATEXMK_FLAGS = -pdf -interaction=nonstopmode -halt-on-error

# Directories
BIBLIOGRAPHY_DIR = ../08_bibliography
TIKZ_BASICS_DIR = ../10_tikz_basics
TIKZ_ADVANCED_DIR = ../11_tikz_advanced
BEAMER_DIR = ../12_beamer
CUSTOM_COMMANDS_DIR = ../13_custom_commands
PROJECTS_DIR = .

# Source files
BIBLIOGRAPHY_SRC = $(BIBLIOGRAPHY_DIR)/bibliography_demo.tex
TIKZ_SHAPES_SRC = $(TIKZ_BASICS_DIR)/tikz_shapes.tex
TIKZ_DIAGRAM_SRC = $(TIKZ_BASICS_DIR)/tikz_diagram.tex
PGFPLOTS_SRC = $(TIKZ_ADVANCED_DIR)/pgfplots_demo.tex
NEURAL_NET_SRC = $(TIKZ_ADVANCED_DIR)/neural_network.tex
PRESENTATION_SRC = $(BEAMER_DIR)/presentation.tex
CUSTOM_MACROS_SRC = $(CUSTOM_COMMANDS_DIR)/custom_macros.tex
ACADEMIC_PAPER_SRC = $(PROJECTS_DIR)/academic_paper.tex

# PDF outputs
BIBLIOGRAPHY_PDF = $(BIBLIOGRAPHY_DIR)/bibliography_demo.pdf
TIKZ_SHAPES_PDF = $(TIKZ_BASICS_DIR)/tikz_shapes.pdf
TIKZ_DIAGRAM_PDF = $(TIKZ_BASICS_DIR)/tikz_diagram.pdf
PGFPLOTS_PDF = $(TIKZ_ADVANCED_DIR)/pgfplots_demo.pdf
NEURAL_NET_PDF = $(TIKZ_ADVANCED_DIR)/neural_network.pdf
PRESENTATION_PDF = $(BEAMER_DIR)/presentation.pdf
CUSTOM_MACROS_PDF = $(CUSTOM_COMMANDS_DIR)/custom_macros.pdf
ACADEMIC_PAPER_PDF = $(PROJECTS_DIR)/academic_paper.pdf

# All PDFs
ALL_PDFS = $(BIBLIOGRAPHY_PDF) $(TIKZ_SHAPES_PDF) $(TIKZ_DIAGRAM_PDF) \
           $(PGFPLOTS_PDF) $(NEURAL_NET_PDF) $(PRESENTATION_PDF) \
           $(CUSTOM_MACROS_PDF) $(ACADEMIC_PAPER_PDF)

# Colors for output
COLOR_RESET = \033[0m
COLOR_GREEN = \033[32m
COLOR_YELLOW = \033[33m
COLOR_BLUE = \033[34m

# Default target: build all PDFs
.PHONY: all
all: $(ALL_PDFS)
	@echo "$(COLOR_GREEN)All LaTeX examples compiled successfully!$(COLOR_RESET)"

# Individual targets
.PHONY: bibliography
bibliography: $(BIBLIOGRAPHY_PDF)

.PHONY: tikz-shapes
tikz-shapes: $(TIKZ_SHAPES_PDF)

.PHONY: tikz-diagram
tikz-diagram: $(TIKZ_DIAGRAM_PDF)

.PHONY: pgfplots
pgfplots: $(PGFPLOTS_PDF)

.PHONY: neural-network
neural-network: $(NEURAL_NET_PDF)

.PHONY: presentation
presentation: $(PRESENTATION_PDF)

.PHONY: custom-macros
custom-macros: $(CUSTOM_MACROS_PDF)

.PHONY: academic-paper
academic-paper: $(ACADEMIC_PAPER_PDF)

# Bibliography compilation (requires biber)
$(BIBLIOGRAPHY_PDF): $(BIBLIOGRAPHY_SRC) $(BIBLIOGRAPHY_DIR)/references.bib
	@echo "$(COLOR_BLUE)Compiling bibliography demo...$(COLOR_RESET)"
	cd $(BIBLIOGRAPHY_DIR) && $(LATEX) bibliography_demo.tex
	cd $(BIBLIOGRAPHY_DIR) && $(BIBER) bibliography_demo
	cd $(BIBLIOGRAPHY_DIR) && $(LATEX) bibliography_demo.tex
	cd $(BIBLIOGRAPHY_DIR) && $(LATEX) bibliography_demo.tex
	@echo "$(COLOR_GREEN)Bibliography demo compiled!$(COLOR_RESET)"

# TikZ basic shapes
$(TIKZ_SHAPES_PDF): $(TIKZ_SHAPES_SRC)
	@echo "$(COLOR_BLUE)Compiling TikZ shapes...$(COLOR_RESET)"
	cd $(TIKZ_BASICS_DIR) && $(LATEX) tikz_shapes.tex
	cd $(TIKZ_BASICS_DIR) && $(LATEX) tikz_shapes.tex
	@echo "$(COLOR_GREEN)TikZ shapes compiled!$(COLOR_RESET)"

# TikZ diagrams
$(TIKZ_DIAGRAM_PDF): $(TIKZ_DIAGRAM_SRC)
	@echo "$(COLOR_BLUE)Compiling TikZ diagrams...$(COLOR_RESET)"
	cd $(TIKZ_BASICS_DIR) && $(LATEX) tikz_diagram.tex
	cd $(TIKZ_BASICS_DIR) && $(LATEX) tikz_diagram.tex
	@echo "$(COLOR_GREEN)TikZ diagrams compiled!$(COLOR_RESET)"

# PGFPlots (may require --shell-escape for some plots)
$(PGFPLOTS_PDF): $(PGFPLOTS_SRC)
	@echo "$(COLOR_BLUE)Compiling PGFPlots demo...$(COLOR_RESET)"
	cd $(TIKZ_ADVANCED_DIR) && $(LATEX) -shell-escape pgfplots_demo.tex
	cd $(TIKZ_ADVANCED_DIR) && $(LATEX) -shell-escape pgfplots_demo.tex
	@echo "$(COLOR_GREEN)PGFPlots demo compiled!$(COLOR_RESET)"

# Neural network visualization
$(NEURAL_NET_PDF): $(NEURAL_NET_SRC)
	@echo "$(COLOR_BLUE)Compiling neural network diagrams...$(COLOR_RESET)"
	cd $(TIKZ_ADVANCED_DIR) && $(LATEX) neural_network.tex
	cd $(TIKZ_ADVANCED_DIR) && $(LATEX) neural_network.tex
	@echo "$(COLOR_GREEN)Neural network diagrams compiled!$(COLOR_RESET)"

# Beamer presentation
$(PRESENTATION_PDF): $(PRESENTATION_SRC)
	@echo "$(COLOR_BLUE)Compiling Beamer presentation...$(COLOR_RESET)"
	cd $(BEAMER_DIR) && $(LATEX) presentation.tex
	cd $(BEAMER_DIR) && $(LATEX) presentation.tex
	@echo "$(COLOR_GREEN)Beamer presentation compiled!$(COLOR_RESET)"

# Custom commands
$(CUSTOM_MACROS_PDF): $(CUSTOM_MACROS_SRC)
	@echo "$(COLOR_BLUE)Compiling custom macros example...$(COLOR_RESET)"
	cd $(CUSTOM_COMMANDS_DIR) && $(LATEX) custom_macros.tex
	cd $(CUSTOM_COMMANDS_DIR) && $(LATEX) custom_macros.tex
	@echo "$(COLOR_GREEN)Custom macros example compiled!$(COLOR_RESET)"

# Academic paper
$(ACADEMIC_PAPER_PDF): $(ACADEMIC_PAPER_SRC)
	@echo "$(COLOR_BLUE)Compiling academic paper...$(COLOR_RESET)"
	cd $(PROJECTS_DIR) && $(LATEX) academic_paper.tex
	cd $(PROJECTS_DIR) && $(LATEX) academic_paper.tex
	cd $(PROJECTS_DIR) && $(LATEX) academic_paper.tex
	@echo "$(COLOR_GREEN)Academic paper compiled!$(COLOR_RESET)"

# Using latexmk (alternative compilation method)
.PHONY: latexmk
latexmk:
	@echo "$(COLOR_YELLOW)Compiling all documents with latexmk...$(COLOR_RESET)"
	cd $(BIBLIOGRAPHY_DIR) && $(LATEXMK) $(LATEXMK_FLAGS) -bibtex bibliography_demo.tex
	cd $(TIKZ_BASICS_DIR) && $(LATEXMK) $(LATEXMK_FLAGS) tikz_shapes.tex
	cd $(TIKZ_BASICS_DIR) && $(LATEXMK) $(LATEXMK_FLAGS) tikz_diagram.tex
	cd $(TIKZ_ADVANCED_DIR) && $(LATEXMK) $(LATEXMK_FLAGS) -shell-escape pgfplots_demo.tex
	cd $(TIKZ_ADVANCED_DIR) && $(LATEXMK) $(LATEXMK_FLAGS) neural_network.tex
	cd $(BEAMER_DIR) && $(LATEXMK) $(LATEXMK_FLAGS) presentation.tex
	cd $(CUSTOM_COMMANDS_DIR) && $(LATEXMK) $(LATEXMK_FLAGS) custom_macros.tex
	cd $(PROJECTS_DIR) && $(LATEXMK) $(LATEXMK_FLAGS) academic_paper.tex
	@echo "$(COLOR_GREEN)All documents compiled with latexmk!$(COLOR_RESET)"

# Clean auxiliary files
.PHONY: clean
clean:
	@echo "$(COLOR_YELLOW)Cleaning auxiliary files...$(COLOR_RESET)"
	find .. -type f \( -name "*.aux" -o -name "*.log" -o -name "*.out" \
	     -o -name "*.toc" -o -name "*.bbl" -o -name "*.blg" -o -name "*.bcf" \
	     -o -name "*.run.xml" -o -name "*.fls" -o -name "*.fdb_latexmk" \
	     -o -name "*.nav" -o -name "*.snm" -o -name "*.vrb" \) -delete
	@echo "$(COLOR_GREEN)Auxiliary files cleaned!$(COLOR_RESET)"

# Clean all generated files including PDFs
.PHONY: cleanall
cleanall: clean
	@echo "$(COLOR_YELLOW)Cleaning all generated files including PDFs...$(COLOR_RESET)"
	find .. -type f -name "*.pdf" -delete
	@echo "$(COLOR_GREEN)All generated files removed!$(COLOR_RESET)"

# Help target
.PHONY: help
help:
	@echo "$(COLOR_BLUE)LaTeX Examples Makefile$(COLOR_RESET)"
	@echo ""
	@echo "Available targets:"
	@echo "  $(COLOR_GREEN)all$(COLOR_RESET)              - Build all LaTeX examples (default)"
	@echo "  $(COLOR_GREEN)bibliography$(COLOR_RESET)     - Build bibliography demo"
	@echo "  $(COLOR_GREEN)tikz-shapes$(COLOR_RESET)      - Build TikZ shapes example"
	@echo "  $(COLOR_GREEN)tikz-diagram$(COLOR_RESET)     - Build TikZ diagrams example"
	@echo "  $(COLOR_GREEN)pgfplots$(COLOR_RESET)         - Build PGFPlots demo"
	@echo "  $(COLOR_GREEN)neural-network$(COLOR_RESET)   - Build neural network visualization"
	@echo "  $(COLOR_GREEN)presentation$(COLOR_RESET)     - Build Beamer presentation"
	@echo "  $(COLOR_GREEN)custom-macros$(COLOR_RESET)    - Build custom macros example"
	@echo "  $(COLOR_GREEN)academic-paper$(COLOR_RESET)   - Build academic paper template"
	@echo "  $(COLOR_GREEN)latexmk$(COLOR_RESET)          - Build all using latexmk (alternative)"
	@echo "  $(COLOR_GREEN)clean$(COLOR_RESET)            - Remove auxiliary files"
	@echo "  $(COLOR_GREEN)cleanall$(COLOR_RESET)         - Remove all generated files including PDFs"
	@echo "  $(COLOR_GREEN)help$(COLOR_RESET)             - Show this help message"
	@echo ""
	@echo "$(COLOR_YELLOW)Requirements:$(COLOR_RESET)"
	@echo "  - pdflatex"
	@echo "  - biber (for bibliography)"
	@echo "  - latexmk (optional, for alternative compilation)"
	@echo ""
	@echo "$(COLOR_YELLOW)Examples:$(COLOR_RESET)"
	@echo "  make                    # Build all examples"
	@echo "  make presentation       # Build only the presentation"
	@echo "  make clean              # Clean auxiliary files"
	@echo "  make cleanall           # Remove all generated files"

# Check if required tools are installed
.PHONY: check
check:
	@echo "$(COLOR_BLUE)Checking for required tools...$(COLOR_RESET)"
	@command -v $(LATEX) >/dev/null 2>&1 || \
	    (echo "$(COLOR_YELLOW)Warning: pdflatex not found$(COLOR_RESET)" && exit 1)
	@echo "  ✓ pdflatex found"
	@command -v $(BIBER) >/dev/null 2>&1 || \
	    echo "$(COLOR_YELLOW)Warning: biber not found (required for bibliography)$(COLOR_RESET)"
	@command -v $(BIBER) >/dev/null 2>&1 && echo "  ✓ biber found"
	@command -v $(LATEXMK) >/dev/null 2>&1 || \
	    echo "$(COLOR_YELLOW)Note: latexmk not found (optional)$(COLOR_RESET)"
	@command -v $(LATEXMK) >/dev/null 2>&1 && echo "  ✓ latexmk found"
	@echo "$(COLOR_GREEN)Tool check complete!$(COLOR_RESET)"

# Quick build (single pass, for testing)
.PHONY: quick
quick:
	@echo "$(COLOR_YELLOW)Quick build (single pass)...$(COLOR_RESET)"
	cd $(BIBLIOGRAPHY_DIR) && $(LATEX) bibliography_demo.tex
	cd $(TIKZ_BASICS_DIR) && $(LATEX) tikz_shapes.tex
	cd $(TIKZ_BASICS_DIR) && $(LATEX) tikz_diagram.tex
	cd $(TIKZ_ADVANCED_DIR) && $(LATEX) -shell-escape pgfplots_demo.tex
	cd $(TIKZ_ADVANCED_DIR) && $(LATEX) neural_network.tex
	cd $(BEAMER_DIR) && $(LATEX) presentation.tex
	cd $(CUSTOM_COMMANDS_DIR) && $(LATEX) custom_macros.tex
	cd $(PROJECTS_DIR) && $(LATEX) academic_paper.tex
	@echo "$(COLOR_GREEN)Quick build complete!$(COLOR_RESET)"

# Watch for changes and rebuild (requires inotify-tools on Linux)
.PHONY: watch
watch:
	@echo "$(COLOR_BLUE)Watching for changes... (Ctrl+C to stop)$(COLOR_RESET)"
	@while true; do \
	    inotifywait -qre modify --include '.*\.tex$$' .. && \
	    make all; \
	done

# Create a distribution tarball
.PHONY: dist
dist: cleanall
	@echo "$(COLOR_BLUE)Creating distribution archive...$(COLOR_RESET)"
	tar czf latex_examples.tar.gz \
	    $(BIBLIOGRAPHY_DIR)/*.tex $(BIBLIOGRAPHY_DIR)/*.bib \
	    $(TIKZ_BASICS_DIR)/*.tex \
	    $(TIKZ_ADVANCED_DIR)/*.tex \
	    $(BEAMER_DIR)/*.tex \
	    $(CUSTOM_COMMANDS_DIR)/*.tex \
	    $(PROJECTS_DIR)/*.tex \
	    $(PROJECTS_DIR)/Makefile
	@echo "$(COLOR_GREEN)Archive created: latex_examples.tar.gz$(COLOR_RESET)"

.PHONY: .SILENT
.SILENT:
