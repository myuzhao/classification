import os
import random
import numpy as np
import soundfile as sf 
import configparser as CP
from datetime import datetime
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import time,h5py,logging,glob,os,argparse
from models.model import TDNN, load_model, save_model
from models.featurizer import AudioFeaturizer

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class TDomainData(Dataset):
    def __init__(self,filepath):
        self.filepath = filepath
        self.f = h5py.File(self.filepath,'r')
        self.dataset_len = self.f['wav_label'].shape[0]

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self,idx):
        if self.f is None:
            self.f = h5py.File(self.filepath,'r')
        wav,label = self.f['wav_label'][idx,:-1], self.f['wav_label'][idx,-1]
        wav,label = wav.squeeze(),label.squeeze()
        return wav,label


class Train():
    def __init__(self, config_file):
        cfg = CP.ConfigParser()
        cfg._interpolation = CP.ExtendedInterpolation()
        cfg.read(config_file)
        self.cfg = cfg

        self.classes_list = self.strToList(cfg['Dataset']['classes_list'])
        self.sample_rate = cfg['Dataset'].getint('sample_rate')
        self.train_path = cfg['Dataset']['train_h5py_file']
        self.test_path = cfg['Dataset']['test_h5py_file']
        self.audio_time = cfg['Dataset'].getint('audio_time')

        self.log_file = cfg['Training']['log_file']
        self.model_prefix = cfg['Training']['model_prefix']
        self.resume_model_flag = cfg['Training'].getboolean('resume_model_flag')
        self.resume_model_path = cfg['Training']['resume_model_path']
        self.train_flag = cfg['Training'].getboolean('train_flag')
        self.batch_size = cfg['Training'].getint('batch_size')
        self.num_workers = cfg['Training'].getint('num_workers')
        self.n_fft = cfg['Training'].getint('n_fft')
        self.f_min = cfg['Training'].getint('f_min')
        self.f_max = cfg['Training'].getint('f_max')
        self.n_mels = cfg['Training'].getint('n_mels')
        #self.spec_width = cfg['Training'].getint('spec_width')
        self.gup_idxs = cfg['Training'].getint('gup_idxs')
        self.learning_rate = cfg['Training'].getfloat('learning_rate')
        self.max_epochs = cfg['Training'].getint('max_epochs')
        self.num_class = len(self.classes_list)
        self.input_size = self.n_mels
        now = datetime.now()
        # 格式化时间字符串，作为文件夹名
        folder_name = now.strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(folder_name, exist_ok=True)
        self.train_folder = folder_name
        shutil.copy(config_file, folder_name)

        logging.basicConfig(level=logging.DEBUG,filename=os.path.join(folder_name, self.log_file),filemode='w+')
        self.set_seed()

    def set_seed(self, seed=42):
        '''Sets the seed of the entire notebook so results are the same every time we run.
        This is for REPRODUCIBILITY.'''
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        logging.debug("SEEDING DONE:{}".format(seed))

    def strToList(self, str):
        str = str.replace("[",'')
        str = str.replace("]",'')
        str = str.replace("'",'')
        str = str.replace(" ",'')
        list = str.split(',')
        return list

    def early_stopping(self):
        #TODO
        pass

    def run(self):
        torch.backends.cudnn.benchmark = True
        train_data = TDomainData(self.train_path)
        train_loader = DataLoader(train_data,batch_size=self.batch_size,shuffle=True,drop_last=True)#,num_workers=self.num_workers,pip_memory=True)

        test_data = TDomainData(self.test_path)
        test_loader = DataLoader(test_data,batch_size=self.batch_size,shuffle=True,drop_last=True)#,num_workers=self.num_workers,pip_memory=True)
        feature_function = AudioFeaturizer(self.sample_rate, self.n_fft, self.n_mels, self.audio_time, self.f_min, self.f_max)
        loss_function = torch.nn.CrossEntropyLoss()
        if self.resume_model_flag:
            model = load_model(self.resume_model_path)
        else:
            model = TDNN(self.num_class, self.input_size)

        model.to(device)
        model = torch.nn.DataParallel(model,device_ids = self.gup_idxs)
        optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate)

        _min,_max = -1, 1
        train_loss_list = []
        test_loss_list = []
        counter = 0
        if (self.train_flag == True):
            for epoch in range(1, self.max_epochs + 1):
                train_loss_all = 0
                train_total_all = 0
                train_correct_all = 0
                model.train()
                start_time = time.time()
                for i_batch,batch_data in enumerate(train_loader):
                    counter += 1
                    wav, label = batch_data
                    wav = wav.float()
                    label = label.to(torch.int64).to(device)
                    one_hot_label = F.one_hot(label, num_classes=self.num_class)
                    feature = feature_function(wav)
                    label_pred = model(feature.to(device).float())

                    _, predicted = torch.max(label_pred.data, 1)
                    correct = (predicted == label).sum().item()
                    train_total_all = train_total_all + label.size(0)
                    train_correct_all = train_correct_all + correct

                    train_loss = loss_function(label_pred.float(),one_hot_label.float())      
                    current_train_loss = train_loss.item()
                    train_loss_all += current_train_loss

                    loss = train_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    for p in model.parameters():
                        p.data.clamp_(_min,_max)

                    if counter%20 == 0:
                        train_acc = train_correct_all / train_total_all
                        avg_train_loss = train_loss_all/counter
                        logging.debug("Epoch {}.......Step: {}/{}.......train loss for epoch: {} {}, train acc:{:.3f}".format(
                                    epoch,counter,len(train_loader), avg_train_loss, current_train_loss, train_acc))

                avg_train_loss = train_loss_all/counter
                train_acc = train_correct_all / train_total_all
                train_loss_list.append(avg_train_loss)
                current_time = time.time()
                
                logging.debug("Epoch {}.......Average train loss for epoch: {}, train acc:{:.3f}".format(epoch, avg_train_loss, train_acc))
                logging.debug("Trian,Totle time elapsed: {} seconds".format(current_time - start_time))
                
                ##test
                with torch.no_grad():
                    test_loss_all = 0
                    test_total_all = 0
                    test_correct_all = 0
                    counter = 0
                    for i_batch,batch_data in enumerate(test_loader):
                        counter += 1
                        wav, label = batch_data
                        wav = wav.to(device).float()
                        label = label.to(torch.int64).to(device)
                        one_hot_label = F.one_hot(label, num_classes=self.num_class)
                        feature = feature_function(wav)
                        label_pred = model(feature.to(device).float())

                        _, predicted = torch.max(label_pred.data, 1)
                        correct = (predicted == label).sum().item()
                        test_total_all = test_total_all + label.size(0)
                        test_correct_all = test_correct_all + correct

                        test_loss = loss_function(label_pred.float(),one_hot_label.float())
                        #loss_function(label_pred.float(),label)
                        test_loss_all += test_loss.item()
                        

                    avg_test_loss = test_loss_all/counter
                    test_acc = test_correct_all / test_total_all
                    logging.debug("Epoch {}.......Average test loss for epoch: {}, test acc:{:.3f}".format(epoch, avg_test_loss, test_acc))
                test_loss_list.append(avg_test_loss)
  
                out_name = "{}_train_loss_{:.3f}_acc_{:.3f}_test_loss_{:.3f}_acc_{:.3f}_out{}.pth".format(self.model_prefix,train_loss.detach().cpu().numpy(), train_acc, test_loss.detach().cpu().numpy(), test_acc, str(epoch)) 
                

                save_model(model,os.path.join(self.train_folder, out_name))



class Infer():
    def __init__(self, config_path):
        cfg = CP.ConfigParser()
        cfg._interpolation = CP.ExtendedInterpolation()
        cfg.read(config_path)
        self.cfg = cfg

        self.sample_rate = cfg['Dataset'].getint('sample_rate')
        self.log_path = cfg['Infer']['log_path']
        self.model_path = cfg['Infer']['infer_path']
        self.infer_flag = cfg['Infer'].getboolean('infer_flag')
        self.gup_idxs = cfg['Training'].getint('gup_idxs')
        self.data_path = cfg['Training']['data_path']

    def run(self):
        if (self.infer_flag == True):
            device = torch.device("cpu")
            model = load_model(args.model_path, args.gpu_idxs)
            model.to(device)
            model.eval()
            infer_path = self.data_path
            print(infer_path)
            for audio_file in glob.glob(os.path.join(infer_path,"*.wav")):
                print(audio_file)
                sn, fs = sf.read(audio_file,dtype = "int16")
                if sn.ndim > 1:
                    sn = sn[:,0]
                sn = sn.astype(np.float32)/32768.0
                sn = sn[np.newaxis,:]
                x = torch.from_numpy(sn)
                label_pred = model(x)
                label_pred = label_pred.detach().cpu().numpy()
                print(label_pred)

def global_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config.cfg',type=str, help='config file')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = global_parse() 
    train = Train(args.config_file)
    train.run()
    #infer = Infer(args.config_file)
    #infer.run()