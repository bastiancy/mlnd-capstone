
Generate pdf with **Pandoc** (based on https://gist.github.com/maxogden/97190db73ac19fc6c1d9beee1a6e4fc8).

```bash
pandoc --filter pandoc-citeproc --bibliography=proposal.bib --variable classoption=onecolumn --variable papersize=a4paper -s proposal.md -o proposal.pdf
```

Use Pandoc with Docker (https://hub.docker.com/r/jpbernius/pandoc/).

```bash
docker run --rm -v `pwd`:/data jpbernius/pandoc -o proposal.pdf proposal.md
docker run --rm -v `pwd`:/data jpbernius/pandoc -H fix-captions.tex --filter pandoc-citeproc --bibliography=proposal.bib --variable classoption=onecolumn --variable papersize=a4paper -s proposal.md -o proposal.pdf
```
