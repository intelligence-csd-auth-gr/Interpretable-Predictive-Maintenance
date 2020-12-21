from IPython.display import display                               
import matplotlib.pyplot as plt
from LioNets import LioNet
from Interpretable_PCA import iPCA
from sklearn.linear_model import Lasso, Ridge, RidgeCV, SGDRegressor
import numpy as np
from ipywidgets import interactive, BoundedFloatText, FloatSlider, IntSlider, ToggleButtons, RadioButtons, \
                       IntRangeSlider, Dropdown, jslink, jsdlink, interactive_output, HBox, VBox, Label

class interactive_iml_tool:
    def __init__(self, predictor, iml_methods, nn_forecasters, dataset, sensors):
        
        """Init function
        Args:
            predictor:
            iml_methods:
            nn_forecasters:
            dataset:
            sensors:
        """
        self.predictor = predictor
        self.lionet = iml_methods['LioNets']
        self.ipca = iml_methods['Interpretable PCA']
        self.forecaster = nn_forecasters['forecaster']
        self.nbeats = nn_forecasters['nbeats']
        self.xyz7_model = nn_forecasters['xyz7_model']
        self.dataset = dataset
        self.sensors = sensors
        
        temp_train = self.dataset.reshape(-1,14)
        self.global_mean, self.global_std = [],[]
        for i in range(14):
            self.global_mean.append(temp_train[:,i].mean())
            self.global_std.append(temp_train[:,i].std())

    def load_instance(self, instance):
        self.instance = instance
        
        # Lionets weights
        model = Ridge(alpha=0.0001,fit_intercept=True,random_state=0)
        lionet_weights, real_prediction, local_prediction = self.lionet.explain_instance(self.instance,200,model)
        
        # Interpretable PCA weights
        _, pca_timestep_weights, _ = self.ipca.find_importance(self.instance)
        pca_timestep_weights = pca_timestep_weights.reshape(700)
        
        self.weights_dict = {'LioNets':lionet_weights,'Interpretable PCA':pca_timestep_weights}
        
        # Original stats
        self.original_instance_statistics = {}
        for method in self.weights_dict.keys():
            self.original_instance_statistics[method] = self.moded_instance_statistics(self.instance,method)
            
        # Recommend modifications
        self.recommendation = {}
        for method in self.weights_dict.keys():
            self.recommendation[method] = self.recommend_modifications(method)  # Lionets & IPCA 
        
        self.seeSens = 1
        self.header = " " 
        self.mod_preds, self.mod_sens_all, self.mod_sens_stats = [],[],[]
        self.main_preds, self.main_sens_all, self.main_sens_stats = [],[],[]
        self.load_UI()
    
    def modify(self, weights, sens, mod, uni_mod_val=0, uni_wght_sign=1, select_rul=0, xyz7_tm=0, forecast_option=6, mod_range=(1,50)):
    
        start, end = mod_range[0], mod_range[1]

        mod_instance = self.instance.copy()
        local_mean = self.instance[start-1:end,sens].mean()

        # ---MODS---        
        if mod == 1: # Uniform
            for i in range(start-1, end):
                if weights.reshape(50,14)[i,sens] > 0 and uni_wght_sign > 0:
                    mod_instance[i,sens] = mod_instance[i,sens] + uni_mod_val
                if weights.reshape(50,14)[i,sens] < 0 and uni_wght_sign < 0:
                    mod_instance[i,sens] = mod_instance[i,sens] + uni_mod_val    
        elif mod == 2: # Local Mean
            mod_instance[start-1:end, sens] = local_mean    
        elif mod == 3: # Global Mean 
            mod_instance[start-1:end, sens] = self.global_mean[sens]   
        elif mod == 4: # Zeros
            mod_instance[start-1:end, sens] = 0.1    
        elif mod == 5: # Gaussian Noise
            for i in range(start-1, end):
                np.random.seed(2000+i)
                gaussian_noise = np.random.normal(self.global_mean[sens], self.global_std[sens], 1)/10
                mod_instance[i,sens] += gaussian_noise[0]
            np.clip(mod_instance,0.1,1.1,out=mod_instance)    
        elif mod == 6: # Neural Forecaster
            prediction = self.forecaster.predict(np.expand_dims(mod_instance,axis=0))
            prediction = prediction.squeeze()
            mod_instance = np.append(mod_instance,prediction,axis=0)
            mod_instance = mod_instance[5:]
        elif mod == 7: # Static Forecaster
            for i in range(mod_instance.shape[1]):
                dif = mod_instance[-1,i] - mod_instance[-6:-1,i]
                temp = np.flip(dif) + mod_instance[-1,i]
                mod_instance[:,i] = np.append(mod_instance[5:,i],temp)  
                np.clip(mod_instance[:,i],0.1,1.1,out=mod_instance[:,i])
        elif mod == 8: # NBeats Forecaster
            prediction = self.nbeats.predict(np.expand_dims(mod_instance,axis=0))
            prediction = prediction.squeeze()
            mod_instance = np.append(mod_instance,prediction,axis=0)
            mod_instance = mod_instance[5:]
        elif mod == 9: # XYZ7 Forecaster
            mod_instance = mod_instance[xyz7_tm*5:45+xyz7_tm*5]
            prediction = self.xyz7_model.predict([np.expand_dims(mod_instance,axis=0),np.array([select_rul])])
            prediction = prediction.squeeze()
            mod_instance = np.append(mod_instance,prediction,axis=0)

        return mod_instance

    def moded_instance_statistics(self, temp_instance, iml_method):
    
        if iml_method == 'Interpretable PCA':
            real_prediction = self.predictor.predict(np.expand_dims(temp_instance,axis=0))
            _, weights, local_prediction  = self.ipca.find_importance(temp_instance)
            weights = weights.reshape(700)
        else:
            model = Ridge(alpha=0.0001,fit_intercept=True,random_state=0)
            weights, real_prediction, local_prediction = self.lionet.explain_instance(temp_instance,200,model)
            weights = weights * temp_instance.reshape(700)

        sensors_all = {}
        count = 0
        for j in range(50):
            count2 = 0
            for i in self.sensors:
                sensors_all.setdefault(i,[]).append([j, weights[count+count2], temp_instance[j][count2],
                                                     weights[count+count2]*temp_instance[j][count2]])
                count2 = count2 + 1
            count = count + 14

        sensors_std = []
        sensors_mean = []
        sensors_max = []
        sensors_min = []
        for i in sensors_all:
            naa = np.array(sensors_all[i])[:,3]
            sensors_std.append(naa.std())
            sensors_mean.append(naa.mean())
            sensors_max.append(naa.max())
            sensors_min.append(naa.min())

        return [real_prediction, local_prediction], sensors_all, [sensors_mean,sensors_std,sensors_min,sensors_max]
    
    def recommend_modifications(self, iml_method):
        
        weights = self.weights_dict[iml_method]
        _, _, original_sens_stats=  self.original_instance_statistics[iml_method]
        sensors_mean =  original_sens_stats[0] #The mean of weights per sensor
        indexed = list(enumerate(sensors_mean))
        indexed.sort(key=lambda tup: tup[1])
        cls0_sens = list([i for i, v in indexed[:2]])
        cls1_sens = list(reversed([i for i, v in indexed[-2:]]))
        #print("Class 0 important sensors:",sensors[cls0_sens[0]], sensors[cls0_sens[1]])
        #print("Class 1 important sensors:",sensors[cls1_sens[0]], sensors[cls1_sens[1]])

        mods = ['Original', 'Uniform', 'Mean(Local)', 'Mean(Global)', 'Zero', \
                'Noise', 'Forecast (Neural)', 'Forecast (Static)', 'Forecast (N-Beats)']
        wghts = ['Negative Weights', 'Positive Weights']

        cls0_mod_results = []
        cls1_mod_results = []
        unif_tests= [0.1, 0.5, -0.1, -0.5]

        for sens in cls0_sens:
            temp = []
            for v,w in zip(unif_tests,np.sign(unif_tests)):
                mod_inst = self.modify(weights, sens, 1, v, w)
                mod_preds = self.predictor.predict(np.array([mod_inst,mod_inst]))[0]
                temp.append((mod_preds[0],sens,1,v,w))
            for mod in range(2,len(mods)):
                mod_inst = self.modify(weights, sens, mod)
                mod_preds = self.predictor.predict(np.array([mod_inst,mod_inst]))[0]
                temp.append((mod_preds[0],sens,mod))
            cls0_mod_results.append(max(temp))

        for sens in cls1_sens:
            temp = []
            for v,w in zip(unif_tests,-np.sign(unif_tests)):
                mod_inst = self.modify(weights, sens, 1, v, w)
                mod_preds = self.predictor.predict(np.array([mod_inst,mod_inst]))[0]
                temp.append((mod_preds[0],sens,1,v,w))
            for mod in range(2,len(mods)):
                mod_inst = self.modify(weights, sens, mod)
                mod_preds = self.predictor.predict(np.array([mod_inst,mod_inst]))[0]
                temp.append((mod_preds[0],sens,mod))
            cls1_mod_results.append(min(temp))


        recommendation = "\t\t\t\t\t\t<<< Recommendations >>>\n\n"
        for e0,rec in enumerate(cls0_mod_results):
            if rec[2]==1:
                recommendation += str(e0+1)+") Try the Uniform modification on sensor "+str(self.sensors[rec[1]])+\
                " with Value: "+str(rec[3])+" on the "+str(wghts[int((1+rec[4])/2)])+" to increase the RUL propability.\n"
            else:
                recommendation += str(e0+1)+") Try the "+str(mods[rec[2]])+" modification on sensor "+str(self.sensors[rec[1]])+ \
                " to increase the RUL propability.\n"

        for e1,rec in enumerate(cls1_mod_results):
            if rec[2]==1:
                recommendation += str(e1+e0+2)+") Try the Uniform modification on sensor "+str(self.sensors[rec[1]])+\
                " with Value: "+str(rec[3])+" on the "+str(wghts[int((1+rec[4])/2)])+" to decrease the RUL propability.\n"
            else:
                recommendation += str(e1+e0+2)+") Try the "+str(mods[rec[2]])+" modification on sensor "+str(self.sensors[rec[1]])+ \
                " to decrease the RUL propability.\n"

        return recommendation
        
    def plot_sensor(self, sens_i, mod_sens_i, mod, rng_sldr, uni_sldr, rd_btn_uni, select_rul, rd_btn_xyz7, forecast_optns, iml_method):

        # Recommend modifications
        print(self.recommendation[iml_method])

        # Disable/Enable UI elements
        if mod>=2 and mod<=6:
            self.mod_settings.children = ([self.opt2_settings])
            self.modify_sens_i.disabled =  False
        elif mod==1:
            self.mod_settings.children = ([self.opt3_settings])
            self.modify_sens_i.disabled =  False
        elif mod==9:
            self.mod_settings.children = ([self.opt4_settings])
            self.forecast.disabled = False if rd_btn_xyz7 else True
            self.modify_sens_i.disabled =  True
        else: 
            self.mod_settings.children = ([self.opt1_settings])
            self.modify_sens_i.disabled =  True

        # If a UI element has been changed other than the Sensor View proceed to the modification
        if self.seeSens == sens_i:
            inst_mod = self.modify(self.weights_dict[iml_method], mod_sens_i-1, mod, uni_sldr, rd_btn_uni,
                              select_rul, rd_btn_xyz7, forecast_optns, rng_sldr)
            self.mod_preds, self.mod_sens_all, self.mod_sens_stats = self.moded_instance_statistics(inst_mod, iml_method)
            if mod==9 and rd_btn_xyz7:
                inst_mod = self.modify(self.weights_dict[iml_method], mod_sens_i-1, forecast_optns)
                self.main_preds, self.main_sens_all, self.main_sens_stats = self.moded_instance_statistics(inst_mod, iml_method)
                self.header = 'EXPECTED'
            else:
                self.main_preds, self.main_sens_all, self.main_sens_stats = self.original_instance_statistics[iml_method]
                self.header = 'ORIGINAL'
        else:
            self.seeSens = sens_i

        # Print the predictions of RUL for the original and modified instance 
        print(self.header+" -> Real prediction: " + str(self.main_preds[0])[:7] + ", Local prediction: " + str(self.main_preds[1])[:7])
        print("  MOD    -> Real prediction: " + str(self.mod_preds[0])[:7] + ", Local prediction: " + str(self.mod_preds[1])[:7])

        # Plotting the figures 
        to_vis = [i[2:] for i in self.sensors]
        x = np.arange(len(to_vis))
        width = 0.4

        fig, axs = plt.subplots(1, 3, figsize=(18, 4), dpi=200)
        axs[0].bar(x-width, self.main_sens_stats[0], width=width, tick_label=to_vis, align='edge', color='C0')
        axs[0].bar(x, self.mod_sens_stats[0], width=width, tick_label=to_vis, align='edge', color='C1')
        axs[0].set_title('Mean')
        axs[0].legend(('Οriginal','Modded'))
        axs[1].bar(x-width, self.main_sens_stats[1], width=width, tick_label=to_vis, align='edge', color='C0')
        axs[1].bar(x, self.mod_sens_stats[1], width=width, tick_label=to_vis, align='edge', color='C1')
        axs[1].set_title('STD')
        axs[2].bar(x-width, self.main_sens_stats[2], width=width, tick_label=to_vis, align='edge', color='C0',)
        axs[2].bar(x-width, self.main_sens_stats[3], width=width, tick_label=to_vis, align='edge', color='C0')
        axs[2].bar(x, self.mod_sens_stats[2], width=width, tick_label=to_vis, align='edge', color='C1')
        axs[2].bar(x, self.mod_sens_stats[3], width=width, tick_label=to_vis, align='edge', color='C1')
        axs[2].set_title('Max and Min')  
        plt.show()

        TIMESTEPS = np.arange(self.instance.shape[0])        
        plt.figure(figsize=(14, 4), dpi=200, facecolor='w', edgecolor='k')
        plt.subplot(131)
        plt.plot(TIMESTEPS,np.array(self.main_sens_all[self.sensors[sens_i-1]])[:,1],color='grey',linestyle = ':')
        plt.plot(TIMESTEPS,np.array(self.mod_sens_all[self.sensors[sens_i-1]])[:,1],color='tab:blue')
        plt.hlines(y=np.array(self.mod_sens_all[self.sensors[sens_i-1]])[:,1].mean(), xmin=0, xmax=50, label='mean')
        plt.title(str("Sensor\'s " + self.sensors[sens_i-1] + " influence"))
        plt.subplot(132)
        plt.plot(TIMESTEPS,np.array(self.main_sens_all[self.sensors[sens_i-1]])[:,2],color='grey',linestyle = ':')
        plt.plot(TIMESTEPS,np.array(self.mod_sens_all[self.sensors[sens_i-1]])[:,2],color='g')
        plt.hlines(y=np.array(self.mod_sens_all[self.sensors[sens_i-1]])[:,2].mean(), xmin=0, xmax=50, label='mean')
        plt.title(str("Sensor\'s " + self.sensors[sens_i-1] + " value"))
        plt.subplot(133)
        plt.plot(TIMESTEPS,np.array(self.main_sens_all[self.sensors[sens_i-1]])[:,3],color='grey',linestyle = ':')
        plt.plot(TIMESTEPS,np.array(self.mod_sens_all[self.sensors[sens_i-1]])[:,3],color='r')
        plt.hlines(y=np.array(self.mod_sens_all[self.sensors[sens_i-1]])[:,3].mean(), xmin=0, xmax=50, label='mean')
        plt.title(str("Sensor\'s " + self.sensors[sens_i-1] + " influence * value"))
        plt.show()
        
    def load_UI(self):
        '''Setting up the interactive visualization tool'''
        
        # UI elements
        range_slider = IntRangeSlider(value=[1,50], min=1, max=50, description="Range: ", continuous_update = False)
        view_sens_i = IntSlider(min=1, max=14, default_value=2, description="View Sensor: ", continuous_update = False)
        self.modify_sens_i = IntSlider(min=1, max=14, default_value=2, description="Mod Sensor: ", continuous_update = False)
        uniform_slider = FloatSlider(value=0, min=-1.1, max=1.1, step=0.05, description='Value:', continuous_update = False)
        radio_button_uni = RadioButtons(options=[('Positive Weights', 1), ('Negative Weights', -1)], description='Affect:')
        select_rul = BoundedFloatText(value=0.5, min=0, max=1, step=0.001, description='Desired RUL:')
        radio_button_xyz7 = RadioButtons(options=[('Present (5-last values)', 0), ('Future (5-next values)', 1)], description='Affect:')
        iml_method = ToggleButtons(options=['LioNets', 'Interpretable PCA'])
        self.forecast = Dropdown(options=[('Neural', 6), ('Static', 7),('N-Beats', 8)], description="Forecast: ")
        mod = Dropdown(options=[('Original', 0), ('Uniform', 1), ('Mean (Local)', 2), ('Mean (Global)', 3), ('Zero', 4), ('Noise', 5),
                                ('Forecast (Neural)', 6), ('Forecast (Static)', 7), ('Forecast (N-Beats)', 8), ('Forecast (XYZ7)', 9)],
                                description="Mods: ")
        jsdlink((self.modify_sens_i, 'value'), (view_sens_i, 'value'))

        # UI layout
        interpretable_settings = HBox([Label('Interpretation method:'), iml_method])
        interpretable_settings.layout.margin = '20px 0 20px 0'
        standard_settings = VBox([self.modify_sens_i,view_sens_i])
        xyz7_settings = VBox([select_rul, radio_button_xyz7])
        xyz7_settings.layout.margin = '0 0 0 20px'
        self.opt1_settings = VBox([mod])
        self.opt2_settings = VBox([mod,range_slider])
        self.opt3_settings = HBox([VBox([mod, range_slider]),VBox([uniform_slider, radio_button_uni])])
        self.opt4_settings = HBox([VBox([mod, self.forecast]),xyz7_settings])
        self.mod_settings = VBox([])
        ui = VBox([interpretable_settings, HBox([standard_settings,self.mod_settings])])

        # Starting the interactive tool
        inter = interactive_output(self.plot_sensor, {'sens_i':view_sens_i, 'mod_sens_i':self.modify_sens_i,'mod':mod,
                                'rng_sldr':range_slider, 'uni_sldr':uniform_slider, 'rd_btn_uni':radio_button_uni, 'select_rul':select_rul, 
                                'rd_btn_xyz7':radio_button_xyz7, 'forecast_optns':self.forecast, 'iml_method':iml_method})
        display(ui,inter)
