import os
import h5py
import random
import pandas as pd
import numpy as np
from multiprocessing import Process,Queue,pool
import soundfile as sf
import configparser as CP
import matplotlib.pyplot as plt

class dataPreProc():
    def __init__(self, cfgpath):
        cfg = CP.ConfigParser()
        cfg._interpolation = CP.ExtendedInterpolation()
        cfg.read(cfgpath)
        self.cfg = cfg
        self.classes_list = self.strToList(cfg['Dataset']['classes_list'])
        self.classes_to_label={}
        for i, c in enumerate(self.classes_list):
            self.classes_to_label[c] = i
        self.data_path = cfg['Dataset']['data_path']
        self.test_rate = cfg['Dataset'].getfloat('test_rate')
        self.train_data_list = cfg['Dataset']['train_data_list']
        self.train_h5py_file = cfg['Dataset']['train_h5py_file']
        self.test_data_list = cfg['Dataset']['test_data_list']
        self.test_h5py_file = cfg['Dataset']['test_h5py_file']

    def generateRandTrainTestList(self):
        file_all = []
        for file in os.listdir(self.data_path):
            data_name = self.data_path + "//" + file
            file_all.append(data_name)
        
        test_rate = self.test_rate
        test_len = int(test_rate * len(file_all))
        random.shuffle(file_all)
        test_file = file_all[:test_len]
        train_file = file_all[test_len:]

        f = open(self.test_data_list,"w")
        for wav_path in test_file:
            f.writelines(wav_path.strip())
            #label
            wav_name = os.path.basename(wav_path)
            label = [(int(i)-1) for i in wav_name.split('.')[0].split('-')]
            #tag = self.classes_list[label[2]]
            f.writelines("--label--" + str(label[2]) + '\n')
        f.close()

        f = open(self.train_data_list,"w")
        for wav_path in train_file:
            f.writelines(wav_path.strip())
            #label
            wav_name = os.path.basename(wav_path)
            label = [(int(i)-1) for i in wav_name.split('.')[0].split('-')]
            #tag = self.classes_list[label[2]]
            f.writelines("--label--" + str(label[2]) + '\n')
        f.close()
        
    def strToList(self, str):
        str = str.replace("[",'')
        str = str.replace("]",'')
        str = str.replace("'",'')
        str = str.replace(" ",'')
        list = str.split(',')
        return list

    def dataStatis(self, classes_list, calsses_all):
        count = {}
        for e in classes_list:     
            count[e] = 0
        for e in calsses_all:
            count[e] = count[e] + 1
        count_list = [count[c] for c in count]
        data = pd.DataFrame({'class':classes_list, 'count': count_list})
        fig = plt.figure(figsize=(6, 7), dpi=70)
        plt.bar(data['class'], data['count'],
                width=0.6, align="center", label="count")
        plt.xlabel('class')
        plt.ylabel('count')
        plt.show()
        fig.savefig("classDistribution.png")

    def generateCsv(self, speech_info, csv_name='data.csv'):
        data_path = speech_info[0]
        data_label = speech_info[1]
        df = pd.DataFrame({'path':data_path,'label':data_label})
        df.to_csv(csv_name, index = False, sep = ',')
    
    def audio2H5py(self, list_name, h5py_file):
        DataConfig = self.cfg['Dataset']
        audio_time = DataConfig.getint('audio_time')
        audio_fs = DataConfig.getint('sample_rate')
        audio_samples = audio_time * audio_fs
        label_samples = 1
        h5py_samples = label_samples + audio_samples
        max_num = DataConfig.getint('max_num') 
        dset_name = 'wav_label'
        f = open(list_name,"r")
        data_path_label_all = f.readlines()
        f.close()
        
        audio_num = len(data_path_label_all)
        if not os.path.exists(h5py_file):
            f = h5py.File(h5py_file,'a')
            h5py_data = f.create_dataset(dset_name,shape=(0,h5py_samples),dtype=np.int16,maxshape=(None,h5py_samples),chunks=(1,h5py_samples)) 
            f.close()

        f = h5py.File(h5py_file,'a',swmr=True)
        
        if dset_name not in f.keys():
            h5py_data = f.create_dataset(dset_name,shape=(0,h5py_samples),dtype=np.int16,maxshape=(None,h5py_samples),chunks=(1,h5py_samples)) 
        h5py_data = f[dset_name]
        
        i = 0
        try:
            for idx in range(audio_num):
                audio_path = data_path_label_all[idx] 
                filename, label = audio_path.strip().split("--label--")
                try:
                    wav_data = sf.read(filename, dtype='int16')
                    if wav_data[1] != audio_fs:
                        #todo resample
                        data = wav_data[0]
                    else:
                        data = wav_data[0]
                    #data audio_samples
                    if(data.ndim > 1):
                        data = data[0]
                    lens = data.shape[0]
                    if(lens < audio_samples):
                        zeros = np.zeros(audio_samples - lens)
                        data = np.concatenate((data,zeros),0)
                    elif(lens > audio_samples):
                        start_idx = np.random.randint(0, lens - audio_samples)
                        data = data[start_idx:start_idx+audio_samples]
                    label = int(label)
                    pass
                except:
                    print("Read error:",filename)
                    continue

                data = data[np.newaxis,:]
                data = data.astype(np.int16)
                num = h5py_data.shape[0]
                shape_new = (num+data.shape[0],h5py_data.shape[1])
                h5py_data.resize(shape_new)
                h5py_data[num:,:-label_samples] = data
                h5py_data[num:,-label_samples] = label
                i = i + 1
                if(i > max_num):
                    f.close()
                    exit(1)
            f.close()
        except:
            f.close()
        

if __name__ == "__main__":
    cfgpath = "config.cfg"
    dataP = dataPreProc(cfgpath)
    dataP.generateRandTrainTestList()

    dataP.audio2H5py(dataP.train_data_list, dataP.train_h5py_file)
    dataP.audio2H5py(dataP.test_data_list, dataP.test_h5py_file)

