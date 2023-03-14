import utils
import json

class DatasetSpliter:
    train = []
    test = []

    def __init__(self) -> None:
        pass

    def get_image_filename(self, id):
        filled_id = str(id).zfill(6)
        return filled_id+'.png'

    def get_label_filename(self,id):
        filled_id = str(id).zfill(6)
        return filled_id+'_ldmks.txt'

    def split(self, size, proportion_train=0.8):
        (a,b) = utils.create_random_int_arrays(int(size*proportion_train), size - int(size*proportion_train))
        self.train = []
        self.test = []
        for index in a:
            self.train.append(self.get_image_filename(index))        
        for index in b:
            self.test.append(self.get_image_filename(index))        

    def save(self, location):
        data = {
            "train": self.train,
            "test":self.test
        }

        file = open(location, "w", encoding="utf-8")
        json.dump(data, file)
        file.close()

    def load(self, location):
        file = open(location, "r", encoding="utf-8")
        object = json.load(file)
        file.close()
        if object["train"]:
            self.train = object["train"]

        if object["test"]:
            self.test = object["test"]

    def get_train_filenames(self):
        return self.train
    
    def get_test_filenames(self):
        return self.test