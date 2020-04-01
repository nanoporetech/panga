.PHONY: install test docs cython


venv: venv/bin/activate
IN_VENV=. ./venv/bin/activate

venv/bin/activate:
	test -d venv || virtualenv venv --prompt '(panga) ' --python=python2
	${IN_VENV} && pip install pip --upgrade
	${IN_VENV} && pip install "setuptools<45"
	${IN_VENV} && pip install distribute --no-binary distribute
	${IN_VENV} && pip install -r install_requires.txt 
	${IN_VENV} && pip install -r requirements.txt; 

install: venv
	${IN_VENV} && python setup.py install

test: venv
	${IN_VENV} && python setup.py nosetests

cython: venv
	${IN_VENV} && python setup.py use_cython


# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

DOCSRC = docs

docs: venv
	${IN_VENV} && pip install sphinx sphinx_rtd_theme sphinx-argparse
	${IN_VENV} && cd $(DOCSRC) && $(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(DOCSRC)/$(BUILDDIR)/html."
	touch $(DOCSRC)/$(BUILDDIR)/html/.nojekyll
