# Create an HTML page of a Jupyter notebook

## STEPS:

1. Copy the aditional files and ipynb files in to this folder.
2. Make the ipynb to rst (See options below)
3. It isn't included, add the appropriate path to `../tutorials.rst`


A. To compile all the 

`make tutorials`

B. And individual file

./ipynb_to_rst.sh <file.ipynb>

## Common Warnings and Errors:

* Kernel 
You might have troubles if with the kernel name,
make sure you defined the env variable of your kernel:
`export NB_KERNEL=<kernel_name>`

* WARNING: Unknown interpreted text role "raw-latex"
Avoid empty spaces in latex inline text.

* WARNING: Duplicate explicit target name: ""
At the end of the line add a double underscore, instead of one.

* WARNING: Explicit markup ends without a blank line; unexpected unindent.
Math section need double space add an extra line.



## Notes:

The inspiration and part of the code
to include the tutorials as html files from
the jupyter notebook tutorials from
seaborn autodocumentation files:

https://github.com/mwaskom/seaborn/blob/master/doc
