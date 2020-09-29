rst_files := $(patsubst %.ipynb,%.rst,$(wildcard *.ipynb))

tutorial: ${rst_files}

%.rst: %.ipynb
	../tools/nb_to_doc.py $*.ipynb
	@cp -r $*_files ../generated/
	@if [ -f ../generated/tequila.$*.rst ]; then \
	    touch ../generated/tequila.$*.rst; \
	 fi

clean:
	rm -rf *.rst
	rm -rf *_files/
	rm -rf .ipynb_checkpoints/
