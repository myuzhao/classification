import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import time,h5py,logging,glob,os,argparse
import numpy as np
import soundfile as sf 
from models.model import TDNN, load_model, save_model
from models.featurizer import AudioFeaturizer
import configparser as CP

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
    def __init__(self, config_path):
        cfg = CP.ConfigParser()
        cfg._interpolation = CP.ExtendedInterpolation()
        cfg.read(config_path)
        self.cfg = cfg

        self.classes_list = self.strToList(cfg['Dataset']['classes_list'])
        self.sample_rate = cfg['Dataset'].getint('sample_rate')
        self.train_path = cfg['Dataset']['train_h5py_file']
        self.test_path = cfg['Dataset']['test_h5py_file']
        self.log_path = cfg['Training']['log_path']
        self.model_prefix = cfg['Training']['model_prefix']
        self.resume_model_flag = cfg['Training'].getboolean('resume_model_flag')
        self.resume_model_path = cfg['Training']['resume_model_path']
        self.train_flag = cfg['Training'].getboolean('train_flag')
        self.batch_size = cfg['Training'].getint('batch_size')
        self.num_workers = cfg['Training'].getint('num_workers')
        self.n_fft = cfg['Training'].getint('n_fft')
        self.n_mels = cfg['Training'].getint('n_mels')
        self.gup_idxs = cfg['Training'].getint('gup_idxs')
        self.learning_rate = cfg['Training'].getfloat('learning_rate')
        self.max_epochs = cfg['Training'].getint('max_epochs')
        self.num_class = len(self.classes_list)
        self.input_size = self.n_mels

    def strToList(self, str):
        str = str.replace("[",'')
        str = str.replace("]",'')
        str = str.replace("'",'')
        str = str.replace(" ",'')
        list = str.split(',')
        return list

    def run(self):
        logging.basicConfig(level=logging.DEBUG,filename=self.log_path,filemode='w+')
        torch.backends.cudnn.benchmark = True
        train_data = TDomainData(self.train_path)
        train_loader = DataLoader(train_data,batch_size=self.batch_size,shuffle=True,drop_last=True)#,num_workers=self.num_workers,pip_memory=True)

        test_data = TDomainData(self.test_path)
        test_loader = DataLoader(test_data,batch_size=self.batch_size,shuffle=True,drop_last=True)#,num_workers=self.num_workers,pip_memory=True)
        feature_function = AudioFeaturizer(self.sample_rate, self.n_fft, self.n_mels, feature_method='MFCC')
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
                    input_lens_ratio = np.array(0.0)
                    feature = feature_function(wav, torch.from_numpy(input_lens_ratio))
                    label_pred = model(feature.to(device).float())

                    _, predicted = torch.max(label_pred.data, 1)
                    correct = (predicted == label).sum().item()
                    train_total_all = train_total_all + label.size(0)
                    train_correct_all = train_correct_all + correct

                    train_loss = loss_function(label_pred.float(),one_hot_label.float())
                    #loss_function(label_pred.float(),label)
                    current_train_loss = train_loss.item()
                    train_loss_all += current_train_loss

                    loss = train_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    for p in model.parameters():
                        p.data.clamp_(_min,_max)

                    if counter%20 == 0:
                        acc = train_correct_all / train_total_all
                        avg_train_loss = train_loss_all/counter
                        logging.debug("Epoch {}.......Step: {}/{}.......train loss for epoch: {} {}, acc:{}".format(
                                    epoch,counter,len(train_loader), avg_train_loss, current_train_loss, acc))

                avg_train_loss = train_loss_all/counter
                acc = train_correct_all / train_total_all
                train_loss_list.append(avg_train_loss)
                current_time = time.time()
                
                logging.debug("Epoch {}.......Average train loss for epoch: {}, acc:{}".format(epoch, avg_train_loss, acc))
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
                        input_lens_ratio = np.array(0.0)
                        feature = feature_function(wav, torch.from_numpy(input_lens_ratio))
                        label_pred = model(feature.to(device).float())

                        _, predicted = torch.max(label_pred.data, 1)
                        correct = (predicted == label).sum().item()
                        test_total_all = test_total_all + label.size(0)
                        test_correct_all = test_correct_all + correct

                        test_loss = loss_function(label_pred.float(),one_hot_label.float())
                        #loss_function(label_pred.float(),label)
                        test_loss_all += test_loss.item()
                        

                    avg_test_loss = test_loss_all/counter
                    acc = test_correct_all / test_total_all
                    logging.debug("Epoch {}.......Average test loss for epoch: {}, acc:{}".format(epoch, avg_test_loss, acc))
                test_loss_list.append(avg_test_loss)
  
                outname = "{}_{}_{}_out{}.pth".format(self.model_prefix,str(train_loss.detach().cpu().numpy()), str(test_loss.detach().cpu().numpy()), str(epoch)) 
                save_model(model,outname)



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
    parser.add_argument('--config_path', default='config.cfg',type=str, help='config path')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = global_parse() 
    train = Train(args.config_path)
    train.run()
    #infer = Infer(args.config_path)
    #infer.run()