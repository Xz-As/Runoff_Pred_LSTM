import sys
import getopt
from pathlib import Path
from typing import Tuple, List
from sklearn.metrics import r2_score
import gcsfs
import pywt
import glob
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
from param_extend import params as pa
from param_extend import m as p_m
import metrics as mat
from camels_api import m as C_m


# Global Var
ARGS = sys.argv[1:]
FILE_SYSTEM = gcsfs.core.GCSFileSystem()
DATA_ROOT = Path(r'.')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # This line checks if GPU is available
areas={'01022500':620.38, '01031500':766.53, '01047000':904.94, '01052500':396.1, '01054200':181.33, '01057000':197.7, '01073000':32.11}



def load_other_datas(basin:str) -> Tuple[pd.DataFrame, int]:
    """Load the meteorological forcing data and the discharge of a specific basin.

    :param basin: 8-digit code of basin as string.

    :return: pd.DataFrame containing the meteorological forcing data and the
        area of the basin as integer.
    """
    # root directory of meteorological forcings
    forcing_path = DATA_ROOT

    # get path of forcing file
    files = list(glob.glob(f"{str(forcing_path)}/{basin}_h*.txt"))
    if len(files) == 0:
        raise RuntimeError(f'No forcing file file found for Basin {basin}')
    else:
        file_path = files[0]

    # read-in data and convert date to datetime index

    df = pd.read_csv(file_path, sep='\s+', header=0)
    return df


def load_forcing_discharge(basin: str) -> Tuple[pd.DataFrame, int]:
    """Load the meteorological forcing data and the discharge of a specific basin.

    :param basin: 8-digit code of basin as string.

    :return: pd.DataFrame containing the meteorological forcing data and the
        area of the basin as integer.
    """
    # root directory of meteorological forcings
    """forcing_path = CAMELS_ROOT / 'basin_mean_forcing' / 'daymet'"""
    forcing_path = DATA_ROOT

    # get path of forcing file
    """files = list(glob.glob(f"{str(forcing_path)}/**/{basin}_*.txt"))"""
    #print(f"{str(forcing_path)}/{basin}_d*.txt")
    files = list(glob.glob(f"{str(forcing_path)}/{basin}_d*.txt"))
    if len(files) == 0:
        raise RuntimeError(f'No forcing file file found for Basin {basin}')
    else:
        file_path = files[0]

    # read-in data and convert date to datetime index

    df = pd.read_csv(file_path, sep='\s+', header=0)
    #print(df)

    nums = (df.N.map(str))
    #print(nums)
    Q_obs = df.Q
    area = areas[basin]
    Q_obs = 28316846.592 * Q_obs * 86400 / (area * 10 ** 6)
    return df, Q_obs, nums



@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix c  ontaining the output feature.
    :param seq_length: Length of look back days for one day of prediction

    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape
    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]
    return x_new, y_new


# ### PyTorch data set

"""dates is N"""
class yiluo_dataset(Dataset):
    """Torch Dataset for basic use of data from the CAMELS data set.

    This data set provides meteorological observations and discharge of a given
    basin from the CAMELS data set.
    """

    def __init__(self, times, basin: str, seq_length: int = 365, period: str = None,
                 dates: List = None, means: pd.Series = None, stds: pd.Series = None,
                 choices = False):
        """Initialize Dataset containing the data of a single basin.

        :param basin: 8-digit code of basin as string.
        :param seq_length: (optional) Length of the time window of
            meteorological input provided for one time step of prediction.
        :param period: (optional) One of ['train', 'eval']. None loads the
            entire time series.
        :param dates: (optional) List of pd.DateTimes of the start and end date
            of the discharge period that is used.
        :param means: (optional) Means of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_means() on the data set.
        :param stds: (optional) Stds of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_stds() on the data set.
        """
        self.time = times
        self.basin = basin
        self.seq_length = seq_length
        self.period = period
        self.dates = dates
        self.means = means
        self.stds = stds
        self.choices = choices

        # load data into memory
        self.x, self.y = self._load_data()

        # store number of samples as class attribute
        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def _load_data(self):
        """Load input and output data from text files."""
        df2, QObs, nums = load_forcing_discharge(self.basin)
        df1 = load_other_datas(self.basin)
        df1['QObs(mm/d)'] = QObs
        for i in df1.keys():
            df2[i] = df1[i]
        #choose the length
        df1 = {}
        if self.dates is not None:
            start_date = self.dates[0]
            if self.dates[0] == self.dates[1]:
                end_date = start_date + len(nums)
                df1 = list(df2[i][start_date : end_date])
            else:
                for i in df2.keys():
                    df1[i] = list(df2[i][start_date : self.dates[1]])
        df = {}
        for i in df1.keys():
            list1 = list(df1[i][1:])
            df[str(i)] = pd.Series(list1)
        list1 = list(QObs[start_date: self.dates[1] - 1])
        df['Q'] = pd.Series(list1)
        df = pd.DataFrame(df)
        # if training period store means and stds
        if self.period == 'train':
            self.means = df.mean()
            self.stds = df.std()

        #print(self.means)

        # extract input and output features from DataFrame
        if self.choices:
            x = np.array([df['P'].values,
                      df['E'].values,
                      df['T'].values,
                      df['Q'].values,
                      #df['EE'].values,
                      #df['WW'].values,
                      #df['RG'].values,
                      #df['R'].values,
                      #df['RS'].values,
                      #df['RSS'].values,
                      #df['EU'].values,
                      #df['EL'].values,
                      #df['ED'].values,
                      #df['WU'].values,
                      #df['WL'].values,
                      #df['WD'].values
                      ]).T
                      #df['srad(W/m2)'].values,
                      #df['tmax(C)'].values,
                      #df['tmin(C)'].values,
                      #df['vp(Pa)'].values]).T
        else:
            x = np.array([df['P'].values,
                      #df['E'].values,
                      #df['T'].values,
                      #df['EE'].values,
                      #df['WW'].values,
                      df['Q'].values,
                      df['RG'].values,
                      #df['R'].values,
                      df['RS'].values,
                      df['RSS'].values,
                      df['EU'].values,
                      df['EL'].values,
                      #df['ED'].values,
                      #df['WU'].values,
                      #df['WL'].values,
                      #df['WD'].values
                      ]).T
                      #df['srad(W/m2)'].values,
                      #df['tmax(C)'].values,
                      #df['tmin(C)'].values,
                      #df['vp(Pa)'].values]).T
        y = np.array([df['QObs(mm/d)'].values]).T

        # normalize data, reshape for LSTM training and remove invalid samples
        x = self._local_normalization(x, variable='inputs')
        x, y = reshape_data(x, y, self.seq_length)
        

        if self.period == "train":
            # Delete all samples, where discharge is NaN
            if np.sum(np.isnan(y)) > 0:
                print(f"Deleted some records because of NaNs {self.basin}")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)

            # Deletes all records, where no discharge was measured (-999)
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)

            # normalize discharge
            y = self._local_normalization(y, variable='output')

        # convert arrays to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        return x, y

    def _local_normalization(self, feature: np.ndarray, variable: str) -> np.ndarray:
        """Normalize input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            # no tmax & tmin
            # srad is the radiation of sun
            if self.choices:
                means = np.array([self.means['P'],
                              self.means['E'],
                              self.means['T'],
                              self.means['Q']
                              #self.means['EE'],
                              #self.means['WW'],
                              #self.means['RG'],
                              #self.means['R'],
                              #self.means['RS'],
                              #self.means['RSS'],
                              #self.means['EU'],
                              #self.means['EL'],
                              #self.means['ED'],
                              #self.means['WU'],
                              #self.means['WL'],
                              #self.means['WD']
                              ])
                stds  =  np.array([self.stds['P'],
                               self.stds['E'],
                               self.stds['T'],
                               self.stds['Q']
                               #self.stds['EE'],
                               #self.stds['WW'],
                               #self.stds['RG'],
                               #self.stds['R'],
                               #self.stds['RS'],
                               #self.stds['RSS'],
                               #self.stds['EU'],
                               #self.stds['EL'],
                               #self.stds['ED'],
                               #self.stds['WU'],
                               #self.stds['WL'],
                               #self.stds['WD']
                               ])
            else:
                means = np.array([self.means['P'],
                              #self.means['E'],
                              #self.means['T'],
                              #self.means['EE'],
                              #self.means['WW'],
                              self.means['Q'],
                              self.means['RG'],
                              #self.means['R'],
                              self.means['RS'],
                              self.means['RSS'],
                              self.means['EU'],
                              self.means['EL'],
                              #self.means['ED'],
                              #self.means['WU'],
                              #self.means['WL'],
                              #self.means['WD']
                              ])
                stds  =  np.array([self.stds['P'],
                               #self.stds['E'],
                               #self.stds['T'],
                               #self.stds['EE'],
                               #self.stds['WW'],
                               self.stds['Q'],
                               self.stds['RG'],
                               #self.stds['R'],
                               self.stds['RS'],
                               self.stds['RSS'],
                               self.stds['EU'],
                               self.stds['EL'],
                               #self.stds['ED'],
                               #self.stds['WU'],
                               #self.stds['WL'],
                               #self.stds['WD']
                               ])
            #print(feature, '\n', means, '\n', stds)
            feature = (feature - means) / stds
        elif variable == 'output':
            feature = ((feature - self.means["QObs(mm/d)"]) /
                       self.stds["QObs(mm/d)"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
        """Rescale input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            if self.choices:
                means = np.array([self.means['P'],
                              self.means['E'],
                              self.means['T'],
                              self.means['Q'],
                              #self.means['EE'],
                              #self.means['WW'],
                              #self.means['RG'],
                              #self.means['R'],
                              #self.means['RS'],
                              #self.means['RSS'],
                              #self.means['EU'],
                              #self.means['EL'],
                              #self.means['ED'],
                              #self.means['WU'],
                              #self.means['WL'],
                              #self.means['WD']
                              ])
                stds  =  np.array([self.stds['P'],
                               self.stds['E'],
                               self.stds['T'],
                               self.stds['Q'],
                               #self.stds['EE'],
                               #self.stds['WW'],
                               #self.stds['RG'],
                               #self.stds['R'],
                               #self.stds['RS'],
                               #self.stds['RSS'],
                               #self.stds['EU'],
                               #self.stds['EL'],
                               #self.stds['ED'],
                               #self.stds['WU'],
                               #self.stds['WL'],
                               #self.stds['WD']
                               ])
            else:
                means = np.array([self.means['P'],
                              #self.means['E'],
                              #self.means['T'],
                              #self.means['EE'],
                              #self.means['WW'],
                              self.means['Q'],
                              self.means['RG'],
                              #self.means['R'],
                              self.means['RS'],
                              self.means['RSS'],
                              self.means['EU'],
                              self.means['EL'],
                              #self.means['ED'],
                              #self.means['WU'],
                              #self.means['WL'],
                              #self.means['WD']
                              ])
                stds  =  np.array([self.stds['P'],
                               #self.stds['E'],
                               #self.stds['T'],
                               #self.stds['EE'],
                               #self.stds['WW'],
                               self.stds['Q'],
                               self.stds['RG'],
                               #self.stds['R'],
                               self.stds['RS'],
                               self.stds['RSS'],
                               self.stds['EU'],
                               #self.stds['EL'],
                               #self.stds['ED'],
                               #self.stds['WU'],
                               #self.stds['WL'],
                               #self.stds['WD']
                               ])
            feature = feature * stds + means
        elif variable == 'output':
            feature = (feature * self.stds["QObs(mm/d)"] +
                       self.means["QObs(mm/d)"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds


class Model(nn.Module):

    def __init__(self, hidden_size: int, dropout_rate: float = 0.01, choices = False):
        """Initialize model

        :param hidden_size: Number of hidden units/LSTM cells
        :param dropout_rate: Dropout rate of the last fully connected
            layer. Default 0.0
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # create required layer
        ip_size = 7
        if choices:
            ip_size = 4
        self.lstm =nn.LSTM(input_size=ip_size, hidden_size=self.hidden_size,
                            num_layers=1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Network.

        :param x: Tensor of shape [batch size, seq length, num features]
            containing the input data for the LSTM network.

        :return: Tensor containing the network predictions
        """
        #print(x.size())
        output, (h_n, c_n) = self.lstm(x)
        """print(output)
        print(h_n)
        print(c_n)"""

        # perform prediction only at the end of the input sequence
        pred = self.fc(self.dropout(h_n[-1, :, :]))
        return pred


def train_epoch(model, optimizer, loader, loss_func):
    """Train model for a single epoch.

    :param model: A torch.nn.Module implementing the LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param loader: A PyTorch DataLoader, providing the trainings
        data in mini batches.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    # set model to train mode (important for dropout)
    model.train()
    # request mini-batch of data from the loader
    for xs, ys in loader:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        #print(xs.size(), ys.size())
        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
    return loss.item()


def eval_model(model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.

    :return: Two torch Tensors, containing the observations and
        model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # push data to GPU (if available)
            xs = xs.to(DEVICE)
            # get model predictions
            y_hat = model(xs)
            obs.append(ys)
            preds.append(y_hat)

    return torch.cat(obs), torch.cat(preds)


def calc_val(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    
    
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator

    #return nse_val
    return [nse_val,
        mat.calc_nse(obs, sim),
        mat.calc_alpha_nse(obs, sim),
        mat.calc_beta_nse(obs, sim),
        mat.calc_fdc_fms(obs, sim),
        mat.calc_fdc_fhv(obs, sim),
        mat.calc_fdc_flv(obs, sim)]


def myloss(y_true, y_pred):
    return torch.sum((torch.exp(y_pred - y_true) - 1) ** 2)


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # return torch.sum(torch.pow(torch.exp(y_pred - y_true) - 1, 2))
        return torch.mean(torch.pow((y_pred - y_true), 2))


def lr_reduce(epoch, lr):
    return lr - lr * 0.1 * int(epoch / 60)


# ## Prepare everything for training
'数据设定'
basins = ('01022500', '01031500', '01047000', '01052500', '01054200', '01057000')
for basin in basins:
  hidden_size = 15  # Number of LSTM cells
  dropout_rate = 0.0  # Dropout rate of the final fully connected Layer [0.0, 1.0]
  learning_rate = 1e-3  # Learning rate used to update the weights
  sequence_length = 100  # Length of the meteorological record provided to the network
  C_m(basin = basin)
  p_m(basin = basin)
  print(f'Parameters:\ntraining batchsize = 512,\nvalidation batchsize = 2048,\ntesting batchsize = 2048,\ndropout rate = {dropout_rate},\nhidden size = {hidden_size},\nsequence length = {sequence_length},\noriginal learning rate = {learning_rate},\nlr decreasing function is: lr = f(lr, epoch) = lr_ori * (1 - 0.1 * int(epoch / 60)),\n\ndevice info:\ndevice = Inter Core i5-8300H CPU @ 2.30GHz 2.30GHz,\nRAM size = 8.00GB(7.86GB available), ')


  if __name__ == "__main__":
    outputs = ['Days\tObs\tPred']
    nse_mx = 0
    print('\nbasin:', basin)
    cal = False
    choice = False
    cal_p = ''
    fig, ax = plt.subplots(figsize=(12, 4))
    if len(ARGS) > 0:
        opts, args = getopt.getopt(ARGS, 'hni:', ['help', 'nn'])
        for opt, val in opts:
            if opt in ['-h', '--help']:
                print("no args for p-lstm or '-n', '--nn' for lstm-only")
            elif opt in ['-n', '--nn']:
                choice = True
            elif opt == '-i':
                cal_p = val
                cal = True
            else:
                print("arg wrong, now the p-lstm model is running")

    n_epochs = [10, 71]  # Number of training epochs
    nses = []
    nses_ = []
    for ep in range(n_epochs[0], n_epochs[1], 20):
        if not choice:
            j = 0
            print(f'P-lstm is running with {ep} total epoch\n')
        else:
            j = 1
            print(f'Lstm is running with {ep} total epoch\n')
        ##############
        # Data set up#
        ##############
        """ 开始导入 """
        # Training data
        start_date = 1
        end_date = 5000
        ds_train = yiluo_dataset('train', basin, seq_length=sequence_length, period="train", dates=[0, int(end_date-start_date)], choices=choice)
        tr_loader = DataLoader(ds_train, batch_size=512, shuffle=True)

        # Validation data. We use the feature means/stds of the training period for normalization
        means = ds_train.get_means()
        stds = ds_train.get_stds()
        start_date1 = 10001
        end_date1 = 11000
        ds_val = yiluo_dataset('val', basin, seq_length=sequence_length, period="eval", dates=[int(start_date1 - start_date), int(end_date1 - start_date)],
                            means=means, stds=stds, choices=choice)
        val_loader = DataLoader(ds_val, batch_size=2048, shuffle=False)

        # Test data. We use the feature means/stds of the training period for normalization
        start_date1 = 11001
        end_date1 = 12600

        dates=[int(start_date1 - start_date), int(end_date1 - start_date)]
        ds_test = yiluo_dataset('test', basin, seq_length=sequence_length, period="eval", dates=[int(start_date1 - start_date), int(end_date1 - start_date)],
                                means=means, stds=stds, choices=choice)
        test_loader = DataLoader(ds_test, batch_size=2048, shuffle=False)

        #########################
        # Model, Optimizer, Loss#
        #########################

        # Here we create our model, feel free
        model = Model(hidden_size=hidden_size, dropout_rate=dropout_rate, choices=choice).to(DEVICE)

        loss_func = My_loss()


        # ## Train the model
        # Now we gonna train the model for some number of epochs. After each epoch we evaluate the model on the validation period and print out the NSE.

        """ 开始训练 """
        nse = 0
        for i in range(ep):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_reduce(i, learning_rate))
            loss = train_epoch(model, optimizer, tr_loader, loss_func)
            obs, preds = eval_model(model, val_loader)
            preds= ds_val.local_rescale(preds.cpu().numpy(), variable='output')
            nse_ = calc_val(obs.cpu().numpy(), preds)
            nse = nse_[0]
            print(f"Epoch {i + 1}, Loss: {loss:.4f}, Validation NSE: {nse:.2f}")


        if cal:
            start_date1 = 0
            means1 = ds_train.get_means()
            stds1 = ds_train.get_stds()
            ds_test = yiluo_dataset(str(cal_p), seq_length=sequence_length, period="eval", dates=[0, 0],
                                means=means1, stds=stds1, choices=choice)
            test_loader = DataLoader(ds_test, batch_size=2048, shuffle=False)
            obs, preds = eval_model(model, test_loader)
            preds = preds.cpu().numpy()
            preds = ds_test.local_rescale(preds, variable='output')
            obs = []
            for i in range(len(preds)):
                if int(preds[i][0]) < 0:
                    preds[i][0] = 0
            for i in range(len(preds)):
                obs.append([0])
        else: 
            """ 评价 """
            # ## Evaluate independent test set
            # Evaluate on test set
            #print(len(test_loader))
            obs, preds = eval_model(model, test_loader)
            preds = preds.cpu().numpy()
            preds = ds_test.local_rescale(preds, variable='output')
            for i in range(len(preds)):
                if float(preds[i][0]) < 0:
                    preds[i][0] = 0

            obs = obs.cpu().numpy()
            nse_ = calc_val(obs, preds)
            str_nse = []
            for i in range(1, 7):
                str_nse.append(str(nse_[i]))
            nses_.append(','.join(str_nse))
            nse = nse_[0]
            nses.append(str(nse))
            if nse_mx < nse:
                nse_mx = nse
                outputs = ['Days\tObs\tPred']
                for i in range(len(preds)):
                    outputs.append(str(i+1)+'\t'+str(obs[i][0])+'\t'+str(preds[i][0]))
            print('NSE:', nse)

    with open(r'./nses_'+basin+'.txt', 'w') as f:
        f.write('\n'.join(nses))
    with open(r'./all_Vals_' + basin + '.csv', 'w') as f:
        f.write('\n'.join(nses_))

    """ 画图 """
    """# Plot results
    
    start_date = ds_test.dates[0]
    end_date = ds_test.dates[1]
    date_range =[str(i) for i in range(start_date1, start_date1 + len(preds))] #end_date1)]

    print(len(obs))
    #if j == 0:
    if not cal:
        ax.plot(date_range, obs, label=f"observation, NSE = {nse:.3f}")
    ax.plot(date_range, preds, label=f"prediction of {n_epochs[j]} epochs")
    ax.legend()
    ax.set_title(f"Basin {basin}")
    # ax.xaxis.set_tick_params(rotation=90)
    ax.set_xlabel("Date")
    ax.set_xticks([])
    _ = ax.set_ylabel("Discharge (mm/d)")"""

    tx = '\n'.join(outputs)
    with open(str(DATA_ROOT) + f'/{basin}_output.txt',"w") as f:
        f.write(tx)
    tx = 'NSE,alpnse,batnse,FMS,FHV,FLV\n'
    for i in range(7):
        nse_[i] = str(nse_[i])
    tx += ','.join(nse_[1:])
    if choice:
        with open(str(DATA_ROOT) + f'/{basin}_Val_lstm.csv',"w") as f:
            f.write(tx)
    else:
        with open(str(DATA_ROOT) + f'/{basin}_Val_plstm.csv',"w") as f:
            f.write(tx)

    #plt.show
    #plt.show()
