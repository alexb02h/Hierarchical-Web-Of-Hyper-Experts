#!/bin/bash
# alternatively, can run the following commands in this order:
#   pdflatex paper.tex
#   biber paper
#   pdflatex paper.tex
#   pdflatex paper.tex
latexmk -pdf paper.tex
