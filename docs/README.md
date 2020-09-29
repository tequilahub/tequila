# Building html documentation

## Libraries

```
pip install sphinxcontrib-napoleon
pip install sphinx &recommonmark 
pip install sphinx-bootstrap-theme
pip install sphinx-rst-builder
pip install sphinx_automodapi
```

## Compilation

### HTML

To build documentation from doc

```
make html
```

It will be made from the configuration file and rst files in `source` directory, and it will create a set of html files on a new folder called `build`.


### Tutorials

To convert tutorials to html and including them in the documentation:

```
export NB_KERNEL=<ipython kernel with tequila>
```
It will indicate which kernel it will be used to run the notebook.

```
make html
```
It will create an html in the `source/tutorials` directory.
Next, you will need to include the path of the newily created 
html into `source/tutorials/index.rst` as a new `.. raw: html`

## Doc references:

https://sphinx-themes.org/

### Examples

The style and code was done with inspiration on the 
documentation

* https://docs.obspy.org
* https://seaborn.pydata.org/





