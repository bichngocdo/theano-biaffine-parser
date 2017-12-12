class Vocab(object):
    def __init__(self):
        self.str2id = None
        self.id2str = None
        self.unk = None

    def init(self, sentences, unk='<UNK>', special_strs=None, cutoff_threshold=0):
        special_strs = special_strs if special_strs else list()
        self.unk = len(special_strs)
        special_strs.append(unk)

        counts = dict()
        for sentence in sentences:
            for item in sentence:
                counts.setdefault(item, 0)
                counts[item] += 1

        self.str2id = dict()
        self.id2str = list()
        id = 0
        for item in special_strs:
            self.str2id[item] = id
            self.id2str.append(item)
            id += 1
        for item, count in counts.iteritems():
            if count >= cutoff_threshold:
                self.str2id[item] = id
                self.id2str.append(item)
                id += 1

    def __getitem__(self, item):
        return self.str2id.get(item, self.unk)

    def __len__(self):
        return len(self.str2id)

    def lookup(self, id):
        return self.id2str[id]
