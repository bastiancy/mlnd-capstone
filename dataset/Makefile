download-glove:
	wget -P . "http://nlp.stanford.edu/data/glove.840B.300d.zip"
	unzip glove.840B.300d.zip -d glove.840B.300d.txt
	rm glove.840B.300d.zip

build-esp:
	python convert.py esp ./conll2002 ./conll2002/esp
	python build_vocab.py ./conll2002/esp
	python build_glove.py ./conll2002/esp

build-ned:
	python convert.py ned ./conll2002 ./conll2002/ned
	python build_vocab.py ./conll2002/ned
	python build_glove.py ./conll2002/ned
