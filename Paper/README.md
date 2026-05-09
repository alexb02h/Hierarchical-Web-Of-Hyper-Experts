# CSCI 693 Research Paper Template

This repository contains:
* A LaTeX template for writing a research paper, based on the ACM small conference format (adapted from the [CCSC Paper Template](https://github.com/ccsc-journal/ccsc-editor/tree/master))
* Supporting scripts for compiling the paper and managing build artifacts
* A `.gitignore` configured for common LaTeX-generated files

The goal is to make it easy to write, build, and version-control your paper as part of your research project.

## Directory Contents

* `figures/` - Subdirectory for example figures (images, tables, charts, etc)
*  `.gitignore` - [sample for LaTeX projects](https://gist.github.com/kogakure/149016)
* `build.sh` - Compile the paper
* `clean.sh` - Remove generated files
* `paper.bib` - Bibliography file
* `paper.tex` - Main LaTeX document
* `preamble.tex` - Defines the document's settings, load packages, and customized formatting
* `README.md` - Essential info about the 693 paper template

## Writing and Version Control Expectations

* Write incrementally and commit changes regularly.
* It is acceptable to commit incomplete drafts, TODOs, and placeholders.
* Generated files should not be committed.
* Review the Research Paper Requirements in Canvas for more details.

The git repository is the authoritative record of your paper's progress.

## Requirements

To compile the paper locally, you will need:
* A [LaTeX distribution](https://www.latex-project.org/get/)
* A full LaTeX install is recommended (includes packages like `biber` for the bibliography)

To check the installation:
```bash
pdflatex --version
biber --version
```

Notes:
* You may edit `.tex` files without any installing anything special, but to compile the files locally and generate `.pdf` output files, you will need LaTeX installed.
* If you do not want to install LaTeX locally, you may edit and compile using [Overleaf](https://www.overleaf.com/). The main files you will be updating are the primary `paper.tex` file, the bibliography `paper.bib`, the compiled output `paper.pdf`, and any figures that have been added (e.g. charts and images such as `.pdf`, `.png`, or `.jpg` files).
* In all cases, remember to make incremental commit updates to your project repository. The git repo is the system of record.

## Bibliography Notes

* References are stored in `paper.bib`.
* Add references using standard BibTeX entries.
* Use `\cite{key}` to cite references in the text.
* The bibliography is rendered using `biblatex` (package included in the preamble) and `biber` (Biblatex's own data backend processor)

Make sure `paper.tex` includes:
```LaTeX
\addbibresource{paper.bib}
...
\printbibliography
```

## Compiling the Paper

You may use the provided build script to compile the paper with the bibliography. From the `paper/` directory, run:
```
./build.sh
```

This script simply runs `latexmk`, which automates the compilation of LaTeX documents:
```
latexmk -pdf paper.tex
```

You can also run the commands separately, in the following order, to compile the `.tex` file with a corresponding `.bib` bibliography ("paper" may be replaced with a different filename if you so choose):

```
pdflatex paper.tex
biber paper
pdflatex paper.tex
pdflatex paper.tex
```

These commands 1) write the citation info, 2) read, process, and write the bibliography data, 3) resolve the citations, and 4) fix cross-references.

The output file will be called `paper.pdf`. You can open and view the file as you would open and view any file, or use:
```
open paper.pdf
```

### Common Issues

If the bibliography does not appear or you are seeing undefined references:
* Make sure that `pdflatex` and `biber` are installed
* Run the full compilation sequence (remember that it is normal to need multiple runs of `pdflatex`)
* Ensure the LaTeX commands for the bibliography are included in the document
* Ensure citations exist in the document


## Cleaning Generated Files

The provided `.gitignore` should ensure that LaTeX-generated files (auxiliary build artifacts) are not included in the git repo (generated files should not be committed to a git repo unless explicitly instructed). However, if you would like to remove these files locally, you may use the provided clean script (I have only included the files that were generated in my tests; you may want to add or remove file extensions). From the `paper/` directory, run:
```
./clean.sh
```
