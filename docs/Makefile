# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    = 
SPHINXBUILD   = sphinx-build
SOURCEDIR     = ./source
BUILDDIR      = ../../tequila_docs

# Put it first so that "make" without argument is like "make help".
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean      to remove genrated output"
	@echo "  html       to make standalone HTML files"
 #@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@touch  $(BUILDDIR)/html/.nojekyll


clean:
	@rm -r $(BUILDDIR)/html/*
	@rm -r $(BUILDDIR)/doctrees/*

#tutorials:
#	make -C source/tutorials

