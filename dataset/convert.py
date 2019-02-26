"""Reformat files from Conll2002 dataset"""

from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lang', required=True, help='The prefix of files, e.g. "esp" for "esp.train.txt"')
parser.add_argument('src', required=True, help='Folder with source files')
parser.add_argument('dest', required=True, help='Folder to save files')


if __name__ == '__main__':
    args = parser.parse_args()

    def convert_words(lang, name):
        Path(args['dest']).mkdir(parents=True, exist_ok=True)

        with Path(args['src'], "{}.{}".format(lang, name)).open() as f:
            words = []
            tags = []

            for line in f:
                data = line.split()
                if len(data) > 0:
                    words.append(data[0])
                    tags.append(data[2])
                else:
                    with Path(args['dest'], lang, "{}.words.txt".format(name)).open(mode='a') as f2:
                        f2.write(" ".join(words) + "\n")
                    with Path(args['dest'], lang, "{}.tags.txt".format(name)).open(mode='a') as f2:
                        f2.write(" ".join(tags) + "\n")
                    words = []
                    tags = []

    convert_words(args['lang'], 'testa')
    convert_words(args['lang'], 'testb')
    convert_words(args['lang'], 'train')
