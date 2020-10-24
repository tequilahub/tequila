# Tequila Documentation

Please note that README.md is a copy of the main README.
Make sure to update it.

## Requirements

Check requirements.txt

## Build

Build the documentation in `tequila/docs` from info in `tequila\docs\source`:

```
make html
```

It will create a set of files in `..\tequila_docs\build`

## Include a new module
 
1. Include the appropriate path to `tequila\docs\source\package\index.rst`. 
Make sure to include a new folder to include automatically
the set of autodoc files and the toctree in

```
<Module name>
===========

.. rubric:: Modules
.. autosummary::
   :toctree: <module>
   :nosignatures:
   
   <set of modules or classes>
```

2. Rebuild the documentation. 
It will create with autosummary and the appropriate template (`tequila\docs\_templates`) a set of rst files.

3. If you want to custumize, e.g. include or exclude some modules/classes/functions/attributes/etc.
you can go ahead and modify the `.rst` file located in the appropriate folder (they are organized by folders based on the names of the modules and/or classes).

4. Make sure you commit the changes the file.

Notes: The set of rst files are not overwritten, if you want to create another template, you will need to
erase that file, and in the next build, it will be created it.

## Build tutorials

Check the REAME file in `tequila\docs\source\jupyter`.

## Syntax

Some useful links:

* Examples of themes:

https://sphinx-themes.org/

## Rebuild github page:

* The documentation is made from the master branch (for now from `dev_doc`).
* You don’t commit changes in the folder `../tequila_docs/build`, since it contains all the htmls.
The htmls are comited into the branch gh-pages, and this is where the webpage lives.

If you want to update the webpage direclty to gh-pages:

0. Create an empty folder at the same level of tequila package called:
`tequila_docs` and `tequila_docs/build`, if you do not have them.

1. Clone the branch gh-pages in the folder in `tequila_docs/build` as `html`.

2. Rebuild the documentation in `tequila/docs`.

3. Include all the files in `tequila_docs\build\html` into the gh-branch.


