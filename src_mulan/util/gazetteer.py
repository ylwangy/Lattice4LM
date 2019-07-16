from .trie import Trie 
import codecs

class Gazetteer:
    def __init__(self, lower):
        self.trie = Trie()
        self.ent2type = {} ## word list to type
        self.ent2id = {"<UNK>":0}   ## word list to id
        self.lower = lower
        self.space = ""

    def enumerateMatchList(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        match_list = self.trie.enumerateMatch(word_list, self.space)
        return match_list

    def insert(self, word_list, source):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        self.trie.insert(word_list)
        string = self.space.join(word_list)
        if string not in self.ent2type:
            self.ent2type[string] = source
        if string not in self.ent2id:
            self.ent2id[string] = len(self.ent2id)

    def searchId(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        string = self.space.join(word_list)
        if string in self.ent2id:
            return self.ent2id[string]
        return self.ent2id["<UNK>"]

    def searchType(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        string = self.space.join(word_list)
        if string in self.ent2type:
            return self.ent2type[string]
        print("Error in finding entity type at gazetteer.py, exit program! String:", string)
        exit(0)

    def size(self):
        return len(self.ent2type)

    def build_gaz_file(self, gaz_file):
        if gaz_file:
            fins = open(gaz_file, 'r').readlines()
            for fin in fins:
                fin = fin.strip().split()[0]
                if fin:
                    self.insert(fin, "one_source")
            print("Load gaz file: ", gaz_file, " total size:", self.size())
        else:
            logging.info("Gaz file is None, load nothing")


    def convertLattice(self, lattice_Ids_forw):
        # print(lattice_Ids_forw[:50])
        lattice_Ids_back = []
        for i in range(len(lattice_Ids_forw)):
            if (i-1 >= 0  and not lattice_Ids_forw[i-1][0][0] == 0) or (i-2 >= 0  and not lattice_Ids_forw[i-2][0][1] == 0) or (i-3 >= 0  and not lattice_Ids_forw[i-3][0][2] == 0):
                tmp = [0,0,0]
                if (i-1 >= 0 and not lattice_Ids_forw[i-1][0][0] == 0):
                    tmp[0] = lattice_Ids_forw[i-1][0][0]
                if (i-2 >= 0 and not lattice_Ids_forw[i-2][0][1] == 0):
                    tmp[1] = lattice_Ids_forw[i-2][0][1]
                if (i-3 >= 0 and not lattice_Ids_forw[i-3][0][2] == 0):
                    tmp[2] = lattice_Ids_forw[i-3][0][2]
                lattice_Ids_back.append([tmp])
                # if (not tmp[0] == 0 )and (not tmp[1] == 0 )and (not tmp[2] == 0):
                #     print('===')
                #     print(gaz_lexicon2[tmp[0]])
                #     print(gaz_lexicon2[tmp[1]])
                #     print(gaz_lexicon2[tmp[2]])
            else:
                lattice_Ids_back.append([[0,0,0]])

        return lattice_Ids_back


    def build_gaz_lexicon(self, input_file, test_file, gaz_lexicon):

        # aa=input()
        data = []
        data_Ids = []
        lattice_Ids = []
        word_list = []
        with codecs.open(input_file, 'r', encoding='utf-8') as fin:
            for line in fin:
              # data.append([])
              # data_Ids.append([])
              lattice_Ids.append([[0,0,0]])
              line_noseg = line.replace(' ', '').strip()
              for i in range(len(line_noseg)):
                word_list.append(line_noseg[i])
              for idx in range(len(line_noseg)):
                matched_entity = self.enumerateMatchList(word_list[idx:])
                for entity in matched_entity:
                    if entity not in gaz_lexicon:
                        gaz_lexicon[entity] = len(gaz_lexicon)
                matched_length = [len(a) for a in matched_entity]
                matched_Id  = [gaz_lexicon[entity] for entity in matched_entity]
                matched_lattice = [0,0,0]
                for ids, lengths in enumerate(matched_length):
                    matched_lattice[lengths - 2] = matched_Id[ids]
                if matched_Id:
                    # data_Ids.append([matched_Id, matched_length])
                    lattice_Ids.append([matched_lattice])
                else:
                    # data_Ids.append([])
                    lattice_Ids.append([[0,0,0]])
                # data.append(matched_entity)
              # data.append([])
              # data_Ids.append([])
              lattice_Ids.append([[0,0,0]])
              word_list = []

        lattice_Ids_back = self.convertLattice(lattice_Ids)
        # aa=input()
        test_data = []
        test_lattice_Ids = []
        test_data_Ids = []
        word_list = []
        with codecs.open(test_file, 'r', encoding='utf-8') as fin:
            for line in fin:
              # test_data.append([])
              test_data_Ids.append([])
              test_lattice_Ids.append([[0,0,0]])
              line_noseg = line.replace(' ', '').strip()
              for i in range(len(line_noseg)):
                word_list.append(line_noseg[i])
              for idx in range(len(line_noseg)):
                matched_entity = self.enumerateMatchList(word_list[idx:])
                for entity in matched_entity:
                    if entity not in gaz_lexicon:
                        gaz_lexicon[entity] = len(gaz_lexicon)
                matched_length = [len(a) for a in matched_entity]
                matched_Id  = [gaz_lexicon[entity] for entity in matched_entity]
                matched_lattice = [0,0,0]
                for ids, lengths in enumerate(matched_length):
                    matched_lattice[lengths - 2] = matched_Id[ids]
                if matched_Id:
                    # test_data_Ids.append([matched_Id, matched_length])
                    test_lattice_Ids.append([matched_lattice])
                else:
                    # test_data_Ids.append([])
                    test_lattice_Ids.append([[0,0,0]])
                # test_data.append(matched_entity)
              # test_data.append([])
              # test_data_Ids.append([])
              test_lattice_Ids.append([[0,0,0]])
              word_list = []
        test_lattice_Ids_back = self.convertLattice(test_lattice_Ids)

        # return data_Ids, lattice_Ids, lattice_Ids_back, test_data_Ids, test_lattice_Ids, test_lattice_Ids_back, gaz_lexicon
        return lattice_Ids, lattice_Ids_back, test_lattice_Ids, test_lattice_Ids_back, gaz_lexicon

    def build_test_gaz_lexicon(self, input_file, gaz_lexicon):

        # aa=input()
        data = []
        data_Ids = []
        lattice_Ids = []
        word_list = []
        with codecs.open(input_file, 'r', encoding='utf-8') as fin:
            for line in fin:
              lattice_Ids.append([[0,0,0]])
              line_noseg = line.replace(' ', '').strip()
              for i in range(len(line_noseg)):
                word_list.append(line_noseg[i])
              for idx in range(len(line_noseg)):
                matched_entity = self.enumerateMatchList(word_list[idx:])
                matched_length = [len(a) for a in matched_entity]
                matched_Id  = [gaz_lexicon[entity] for entity in matched_entity]
                matched_lattice = [0,0,0]
                for ids, lengths in enumerate(matched_length):
                    matched_lattice[lengths - 2] = matched_Id[ids]
                if matched_Id:
                    lattice_Ids.append([matched_lattice])
                else:
                    lattice_Ids.append([[0,0,0]])
              lattice_Ids.append([[0,0,0]])
              word_list = []

        lattice_Ids_back = self.convertLattice(lattice_Ids)
       
        return lattice_Ids, lattice_Ids_back