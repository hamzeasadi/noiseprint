"""
general utility functions
"""

import os
import pickle
from typing import Any
from dataclasses import dataclass


@dataclass
class Paths:

    project_root:str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    data:str = os.path.join(project_root, "data")
    config:str = os.path.join(project_root, "noiseprint", "config")
    
    model:str = os.path.join(data, "model")

    report:str = os.path.join(data, "report")

    dataset:str = os.path.join(data, "dataset")
    train:str = os.path.join(data, "train")
    valid:str = os.path.join(data, "valid")

    @staticmethod
    def crtdir(path:str):
        if not os.path.exists(path):
            os.makedirs(path)




def crtdir(path:str):
        if not os.path.exists(path):
            os.makedirs(path)




def load_pickle(file_path:str):
    with open(file_path, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    return data




def save_as_pickle(file_name:str, file_path:str, data:Any):
    crtdir(file_path)
    save_path:str = os.path.join(file_path, file_name)
    with open(save_path, "wb") as pickle_file:
        pickle.dump(obj=data, file=pickle_file)









def main():
    """docs"""

    paths = Paths()
    for path in paths.__dict__:
        print(paths.__dict__[path])




if __name__ == "__main__":

    main()