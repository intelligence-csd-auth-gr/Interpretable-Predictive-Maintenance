import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class Load_Dataset:
    """Class for loading preprocessed datasets"""

    def rul_finder(mapper):
        RULs = []
        RULsInd = []
        cnt = 1
        for m in mapper:
            RULsInd.append(cnt)
            cnt += 1
            RULs.append(m[1])
        return RULsInd, RULs

    def load_data_turbofan(plot_RULs=False):
        feature_names = ['u', 't', 'os_1', 'os_2', 'os_3'] #u:unit, t:time, s:sensor
        feature_names += ['s_{0:02d}'.format(s + 1) for s in range(26)]
        fd = {}
        for i in range(4):
            p = 'datasets/CMAPSSData/train_FD00'+ str(i+1) +'.txt'
            df_train = pd.read_csv(p, sep= ' ', header=None, names=feature_names, index_col=False)
            mapper = {}
            for unit_nr in df_train['u'].unique():
                mapper[unit_nr] = df_train['t'].loc[df_train['u'] == unit_nr].max()#max time einai to rul tou
            # calculate RUL = time.max() - time_now for each unit
            df_train['RUL'] = df_train['u'].apply(lambda nr: mapper[nr]) - df_train['t']

            p = 'datasets/CMAPSSData/test_FD00'+ str(i+1) +'.txt'
            df_test = pd.read_csv(p, sep= ' ', header=None, names=feature_names, index_col=False)
            p = 'datasets/CMAPSSData/RUL_FD00'+ str(i+1) +'.txt'
            df_RUL = pd.read_csv(p, sep= ' ', header=None, names=['RUL_actual'], index_col=False)
            temp_mapper = {}
            for unit_nr in df_test['u'].unique():
                temp_mapper[unit_nr] = df_test['t'].loc[df_test['u'] == unit_nr].max()#max time einai to rul tou

            mapper_test = {}
            cnt = 1
            for mt in df_RUL.values:
                mapper_test[cnt]=mt[0]+temp_mapper[cnt]
                cnt += 1
            df_test['RUL'] = df_test['u'].apply(lambda nr: mapper_test[nr]) - df_test['t']

            if plot_RULs:
                mapper = sorted(mapper.items(), key=lambda kv: kv[1])
                plt.figure(figsize=(10, 5))
                ax1 = plt.subplot(121)
                ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
                RULsInd, RULs = rul_finder(mapper)
                ax1.plot(RULsInd, RULs)
                ax1.set_title('Fault Mode '+ str(i+1)+': Train set')
                ax1.set_xlabel('Unit_id')
                ax1.set_ylabel('RUL')  
                mapper_test = sorted(mapper_test.items(), key=lambda kv: kv[1])
                ax2 = plt.subplot(122)
                ax2.margins(0.05)           # Default margin is 0.05, value 0 means fit
                tRULsInd, tRULs = rul_finder(mapper_test)
                ax2.plot(tRULsInd,tRULs)
                ax2.set_title('Fault Mode '+ str(i+1)+': Test set')
                ax2.set_xlabel('Unit_id')
                ax2.set_ylabel('RUL')
                plt.show()
                print('[FaultMode'+ str(i+1) +']','Train Min:',RULs[0],' Max:',RULs[-1],'| Test Min:',tRULs[0],' Max',tRULs[-1])

            s = 'FaultMode'+ str(i+1) +''
            fd[s] = {'df_train': df_train, 'df_test': df_test}
        feature_names.append('RUL')
        return fd, feature_names