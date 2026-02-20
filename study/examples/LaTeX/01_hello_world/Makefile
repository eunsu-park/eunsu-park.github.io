# ============================================================================
# Makefile for LaTeX Compilation
# ============================================================================
# This Makefile automates the compilation of LaTeX documents to PDF.
#
# Usage:
#   make          - Compile hello.tex to hello.pdf
#   make clean    - Remove all generated files
#   make view     - Open the PDF (macOS/Linux)
# ============================================================================

# Variables
LATEX = pdflatex
MAIN = hello
SOURCE = $(MAIN).tex
OUTPUT = $(MAIN).pdf

# Auxiliary files created during compilation
AUX_FILES = *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

# Default target
all: $(OUTPUT)

# Compile the LaTeX document
# Run twice to resolve cross-references and TOC
$(OUTPUT): $(SOURCE)
	@echo "Compiling $(SOURCE)..."
	$(LATEX) $(SOURCE)
	@echo "Running second pass for cross-references..."
	$(LATEX) $(SOURCE)
	@echo "Compilation complete: $(OUTPUT)"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f $(AUX_FILES)
	@echo "Clean complete."

# Clean everything including PDF
cleanall: clean
	@echo "Removing PDF output..."
	rm -f $(OUTPUT)
	@echo "All files removed."

# Open PDF viewer (platform-specific)
view: $(OUTPUT)
ifeq ($(shell uname), Darwin)
	@open $(OUTPUT)
else ifeq ($(shell uname), Linux)
	@xdg-open $(OUTPUT) 2>/dev/null || evince $(OUTPUT) 2>/dev/null
else
	@echo "Please open $(OUTPUT) manually."
endif

# Force recompilation
rebuild: cleanall all

# Help message
help:
	@echo "Available targets:"
	@echo "  make          - Compile LaTeX document"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make cleanall - Remove all generated files including PDF"
	@echo "  make view     - Open PDF in viewer"
	@echo "  make rebuild  - Clean and recompile"

# Phony targets (not actual files)
.PHONY: all clean cleanall view rebuild help

# ============================================================================
# Notes:
# - pdflatex is run twice because cross-references, TOC, etc. need two passes
# - For documents with bibliography or complex references, you may need:
#   pdflatex -> bibtex -> pdflatex -> pdflatex
# - latexmk is a more advanced tool that handles this automatically
# ============================================================================
