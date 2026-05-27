# Minimal Makefile for Sphinx documentation.
# Mirrors the standard sphinx-quickstart output.

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help clean html livehtml linkcheck Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"
	rm -rf "$(SOURCEDIR)"/api/generated
	rm -rf "$(SOURCEDIR)"/api/*/generated
	rm -rf "$(SOURCEDIR)"/_thumbnails
	rm -f  "$(SOURCEDIR)"/examples/gallery.rst
	rm -rf "$(SOURCEDIR)"/kernels/img
	rm -f  "$(SOURCEDIR)"/kernels/gallery.rst

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

linkcheck:
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
