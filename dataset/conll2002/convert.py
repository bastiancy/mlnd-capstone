from pathlib import Path


def convert_words(folder, lang, name):
    with Path(folder, "{}.{}".format(lang, name)).open() as f:
        words = []
        tags = []

        for line in f:
            data = line.split()
            if len(data) > 0: 
                words.append(data[0])
                tags.append(data[2])
            else:
                with Path(folder, lang, "{}.words.txt".format(name)).open(mode='a') as f2:
                    f2.write(" ".join(words) + "\n")
                with Path(folder, lang, "{}.tags.txt".format(name)).open(mode='a') as f2:
                    f2.write(" ".join(tags) + "\n")
                words = []
                tags = []


if __name__ == '__main__':
    convert_words('.', 'esp', 'testa')
    convert_words('.', 'esp', 'testb')
    convert_words('.', 'esp', 'train')
    convert_words('.', 'ned', 'testa')
    convert_words('.', 'ned', 'testb')
    convert_words('.', 'ned', 'train')
