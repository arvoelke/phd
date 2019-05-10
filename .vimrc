autocmd BufWritePost *.tex !latexmk -pdf -pdflatex="pdflatex -shell-escape" phd
