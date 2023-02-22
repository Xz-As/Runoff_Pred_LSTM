from math import exp
from pandas import read_csv as csv
from pandas import to_datetime
from numpy import zeros

def _min(a:int = 0, b:int = -1):
    if a < b:
        return a
    return b


def _abs(a:int = 0):
    if a < 0:
        return -a
    return a


class Camels_basin():
    """
        Loading the data of Camels Dataset and Processing the data
    """
    def __init__(self, Root, Basin:str = '01022500'):
        self.basin = Basin
        self.root = Root
        self.datas, self.area = self.load_data()
        self.Qobs, self.QObs_ori = self.load_discharge()
        self.ln_ = len(self.datas)
        #print(self.ln_)
        self.P_s = []
        self.E_s = []
        self.T_s = []
        self.Q_s = []
        self.S = self.Qobs[self.Qobs.keys()[0]]
        self.S0 = self.S
        


    """preparation"""
    def load_pra(self, i:str = '', ii:str = ''):
        if ii == '':
            self.Q_ori = self.QObs_ori[i]
            if self.Q_ori < 0:
                self.Q_ori = 0
            self.T = ((self.datas['tmax(C)'][i] + self.datas['tmin(C)'][i]) / 2)
            self.P = self.cal_P(self.datas['prcp(mm/day)'][i])
            self.E = 0
            return True
        self.Q = self.Qobs[ii]
        self.Q_ori = self.QObs_ori[i]
        if self.Q_ori < -100:
            self.ln_ -= 1
            return False
        if self.Q_ori < 0:
            self.Q = 0
            self.Q_ori = 0
        self.M = self.datas['swe(mm)'][i]
        self.T = ((self.datas['tmax(C)'][i] + self.datas['tmin(C)'][i]) / 2)
        self.P = self.cal_P(self.datas['prcp(mm/day)'][i])
        self.L_day = self.datas['dayl(s)'][i]
        self.PET = self.cal_PET()
        #self.E = 1
        #self.S = 0
        #print(self.P)
        #j = 0
        #de = 1e-2
        #dd = 1
        #epochs = 1000
        #while de < _abs(dd):
        #for k in range(epochs):
            #if k == 1000 and self.E < 0.00000001:
                #return
            #j += 1
        self.cal_all()
        #print(i, self.E)
        return True
        #return j


    """cal_"""
    def cal_all(self):        
        #self.S = self.cal_s()
        self.E = self.cal_E()


    def cal_P(self, P_):
        if self.T > -1:
            return P_
        return P_#0


    def cal_s(self):
        return self.Q + _min(self.P, self.T - 1) - self.M - self.E# - self.Q


    def cal_E(self):
        if self.S < 0:
            return 0
        elif self.S > 2000:
            return self.PET
        return self.PET / 2000 * self.S


    def cal_PET(self):
        return 29.8 * self.L_day * self.e_() / (self.T + 273.2)


    def e_(self):
        return 0.611 * exp(17.3 * self.T / (self.T + 237.3))


    """ loading data"""
    def load_discharge(self):
      for Num_file in range(1, 19):
        if Num_file < 10:
            f_root = '0'+str(Num_file)
        else :
            f_root = str(Num_file)
        path_ = self.root + r'/usgs_streamflow'  + '/' + f_root + '/' + self.basin + '_streamflow_qc.txt'
        try:
           col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
           df = csv(path_, sep='\s+', header=None, names=col_names)
           
        except:
           if f_root == '18' :
               raise RuntimeError(f"undefined basin {self.basin}")
           continue
        datas = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
        df.index = to_datetime(datas, format="%Y/%m/%d")        
        
        
        dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
        df.index = to_datetime(dates, format="%Y/%m/%d")

        # normalize discharge from cubic feed per second to mm per day
        df['QObs1'] = df['QObs']
        df.QObs1 = 28316846.592 * df.QObs * 86400 / (self.area * 10**6)

        return df.QObs1, df.QObs

    
    def load_data(self):
      for i in range(1, 19):
        if i < 10:
          j = '0' + str(i)
        else:
          j = str(i)
        path_ = self.root + r'/basin_mean_forcing/daymet/' + j + '/' + self.basin + '_lump_cida_forcing_leap.txt'
        try:
           df = csv(path_, sep='\s+', header=3)
           
        except:
           if j == '18' :
             #print(basin, "isn't in dataset\n")
             raise RuntimeError(f"undefined basin {self.basin}")
           continue
        datas = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
        df.index = to_datetime(datas, format="%Y/%m/%d")
        with open(path_, 'r') as fp:
            content = fp.readlines()
            area = int(content[2])

        return df, area


    def saving_data(self, i):
        """self.P_s[i] = self.P
        self.E_s[i] = self.E
        self.T_s[i] = self.T
        self.Q_s[i] = self.Q_ori"""
        self.P_s.append(self.P)
        self.E_s.append(self.E)
        self.T_s.append(self.T)
        self.Q_s.append(self.Q_ori)

    
    def writing_data(self):
        """
        self.P_s = list(self.P_s)
        self.E_s = list(self.E_s)
        self.T_s = list(self.T_s)
        self.Q_s = list(self.Q_s)
        """
        tx = 'N\tP\tE\tT\tQ\n'
        for i in range(len(self.P_s)):
            tx += str(i) + '\t' + str(self.P_s[i]) + '\t' + str(self.E_s[i]) + '\t' + str(self.T_s[i]) + '\t' + str(self.Q_s[i]) + '\n'
        with open('./'+self.basin+'_data.txt', 'w') as f:
            f.write(tx)


def m(basin):
    a = Camels_basin(Root = r'/media/asxz/新加卷/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2', Basin=basin)
    j = 0
    ii = ''
    for i in a.datas['Year'].keys():
        if a.load_pra(i, ii):
            a.saving_data(j)
            j += 1
            ii = i
    a.writing_data()


if __name__ == '__main__':
    m('01022500')
