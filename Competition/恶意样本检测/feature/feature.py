import os
import pefile
import pandas as pd

class PEFile:
    def __init__(self, filename):
        self.pe = pefile.PE(filename, fast_load=True)
        self.filename = filename
        self.DebugSize = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[6].Size
        self.DebugRVA =self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[6].VirtualAddress
        self.ImageVersion = self.pe.OPTIONAL_HEADER.MajorImageVersion
        self.OSVersion = self.pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
        self.ExportRVA = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[0].VirtualAddress
        self.ExportSize = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[0].Size
        self.IATRVA = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[12].VirtualAddress
        self.ResSize = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[2].Size
        self.LinkerVersion = self.pe.OPTIONAL_HEADER.MajorLinkerVersion
        self.NumberOfSections = self.pe.FILE_HEADER.NumberOfSections
        self.StackReserveSize =self.pe.OPTIONAL_HEADER.SizeOfStackReserve
        self.Dll =self.pe.OPTIONAL_HEADER.DllCharacteristics

    def Construct(self):
        sample = {}
        for attr, k in self.__dict__.items():
            if(attr != "pe"):
                sample[attr] = k
        return sample

def pe2vec(direct):
    dataset = {}
    
    for subdir, dirs, files in os.walk(direct):
        for f in files:
            file_path = os.path.join(subdir, f)
            try:
                pe = PEFile(file_path)
                dataset[str(f)] = pe.Construct()
            except Exception as e:
                print(e)
    return dataset

def vec2csv(dataset):
    df = pd.DataFrame(dataset)
    infected = df.transpose()
    infected.to_csv('dataset.csv', sep=',', encoding='utf-8')

if __name__ == '__main__':
    direct = './data/'
    
    dataset = pe2vec(direct)
    vec2csv(dataset)
