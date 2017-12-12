import numpy as np

from conll import CoNLLFile
from vocab import Vocab


class DataLoader(object):
    def __init__(self):
        self.word_vocab = None
        self.pretrain_vocab = None
        self.tag_vocab = None
        self.label_vocab = None

        self.cutoff_threshold = 0
        self.labeled = True
        self.lowercase = False

        self.NONE = '-NONE-'
        self.ROOT = '-ROOT-'
        self.UNKNOWN = '-UNKNOWN-'

    def read_from_file(self, path):
        with open(path, 'r') as f:
            return self.read_from_buffer(f, buffer_size=None)

    def read_from_buffer(self, f, buffer_size=None):
        all_words = list()
        all_tags = list()
        all_heads = list()
        all_labels = list()
        conll_file = CoNLLFile(f)
        k = 0
        for block in conll_file:
            k += 1
            words = list()
            tags = list()
            heads = list()
            labels = list()
            for line in block:
                items = line.rstrip().split('\t')
                if len(items) < 10:
                    print line
                    raise Exception('Malformed CONLL file: wrong number of fields: ' + str(len(items)))
                if self.lowercase:
                    items[1] = items[1].lower()
                words.append(items[1])
                tags.append(items[4])
                heads.append(int(items[6]))
                labels.append(items[7])
            all_words.append(words)
            all_tags.append(tags)
            all_heads.append(heads)
            all_labels.append(labels)
            if buffer_size is not None and k == buffer_size:
                break
        return all_words, all_tags, all_heads, all_labels

    def _convert_to_ids(self, sentences, vocab, level=0):
        if level == 0:
            return vocab[sentences]
        elif level == 1:
            return [vocab[item] for item in sentences]
        elif level == 2:
            return [[vocab[item] for item in sentence] for sentence in sentences]
        else:
            raise Exception('List depth not supported')

    def convert_to_ids(self, words, tags, heads, labels):
        results = list()
        results.append(self._convert_to_ids(words, self.word_vocab, level=2))
        if self.pretrain_vocab:
            results.append(self._convert_to_ids(words, self.pretrain_vocab, level=2))
        results.append(self._convert_to_ids(tags, self.tag_vocab, level=2))
        results.append(heads)
        results.append(self._convert_to_ids(labels, self.label_vocab, level=2))
        return results

    def read_pretrained_embeddings(self, fp):
        str2id = dict()
        id2str = list()
        embeddings = list()
        id = 0
        for str in [self.NONE, self.ROOT, self.UNKNOWN]:
            str2id[str] = id
            id2str.append(str)
            id += 1

        with open(fp, 'r') as f:
            vocab_size, dim = f.readline().rstrip().split()
            print 'Read embeddings from file %s: vocab size is %s, dimension is %s' % (fp, vocab_size, dim)
            for i, line in enumerate(f):
                line = line.rstrip().split()
                try:
                    str2id[line[0]] = id
                    id2str.append(line[0])
                    embeddings.append(line[1:])
                    id += 1
                except:
                    raise ValueError('The embedding file is misformatted at line %d' % (i + 2))
            embeddings = np.array(embeddings, dtype='float32')
            embeddings = np.pad(embeddings, ((3, 0), (0, 0)), 'constant', constant_values=0)

            self.pretrain_vocab = Vocab()
            self.pretrain_vocab.str2id = str2id
            self.pretrain_vocab.id2str = id2str
            self.pretrain_vocab.unk = 2
            return embeddings

    def init_and_load(self, raw_data):
        words, tags, heads, labels = raw_data

        self.word_vocab = Vocab()
        self.tag_vocab = Vocab()
        self.label_vocab = Vocab()

        self.word_vocab.init(words,
                             unk=self.UNKNOWN, special_strs=[self.NONE, self.ROOT],
                             cutoff_threshold=self.cutoff_threshold)
        self.tag_vocab.init(tags,
                            unk=self.UNKNOWN, special_strs=[self.NONE, self.ROOT])
        self.label_vocab.init(labels,
                              unk=self.UNKNOWN)

        return self.convert_to_ids(words, tags, heads, labels)

    def load(self, raw_data):
        words, tags, heads, labels = raw_data
        return self.convert_to_ids(words, tags, heads, labels)
