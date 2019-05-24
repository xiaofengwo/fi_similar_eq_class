class SimilaritySet:
    def __init__(self):
        self.de_set = set()
        self.id = None

    def add(self, de):
        self.de_set.add(de)
        de.belongs_to = self


class DefiniteEquvalanceSet:
    def __init__(self, dyn, static, register, bit, length, fi_result, belongs_to=None):
        self.dyn = dyn
        self.static = static
        self.register = register
        self.bit = bit
        self.length = length
        self.fi_result = fi_result
        if belongs_to is None:
            ss = SimilaritySet()
            ss.add(self)
            self.belongs_to = ss
        self.belongs_to = belongs_to

        self.id = str(self.dyn) + ',' + str(self.static) + ',' + str(self.register)
