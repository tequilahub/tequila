# Tequila Documentation

Please note that README.md is a copy of the main README.
Make sure to update it.


## Dependencies
Check requirements.txt

## Build

build documentation from specs in source directory:

make html

## Include a new module
 
Include the appropriate path to `package/index` 
make sure to include a new folder to include automatically
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

It will create with autosummary and the appropriate template (template folder) a set of rst files.
If you want to custumize, e.g. include or exclude some modules/classes/functions/attributes/etc.
You can go ahead and modifiy the `.rst` file located in the appropriate folder.
Make sure you commit the changes the file.

The set of rst files are not overwritten, if you want to create another template, you will need to
erase that file, and in the next build, it will be built.

## Build tutorials

Check the readme file in jupyter.

## Syntax

Some useful links:

* Examples of themes:

https://sphinx-themes.org/

## Workflow:

* The documentation is made from the master branch.
* You don’t commit changes in the folder doc/build, since it contains all the html
The htmls are comited into the branch gh-pages, and this is where the webpage lives.


