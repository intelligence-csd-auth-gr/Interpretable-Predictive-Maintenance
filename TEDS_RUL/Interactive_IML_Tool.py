from IPython.display import display                               
import matplotlib.pyplot as plt
from lionets import LioNets
from Interpretable_PCA import iPCA
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import Lasso, Ridge, RidgeCV, SGDRegressor
import numpy as np
from itertools import product
from collections import OrderedDict
from ipywidgets import interactive, BoundedFloatText, FloatSlider, IntSlider, ToggleButtons, RadioButtons, Checkbox, \
                       IntRangeSlider, Dropdown, jslink, jsdlink, interactive_output, HBox, VBox, Label

class interactive_iml_tool:
    def __init__(self, predictor, iml_methods, nn_forecasters, data, target, features, target_scaler):
        
        """Init function
        Args:
            predictor:
            iml_methods:
            nn_forecasters:
            data:
            target:
            features:
            scaler:
        """
        self.lionet = iml_methods['LioNets']
        self.lime = LimeTabularExplainer(training_data=data.reshape(len(data),-1), 
                                         discretize_continuous=False, mode="regression",random_state=0)
        self.ipca = {}
        self.ipca['LioNets'] = iPCA(self.lionet.give_me_the_neighbourhood, 'local')
        self.ipca['Lime'] = iPCA(self.lime.give_me_the_neighbourhood, 'local', self._lime_predict)

        self.predictor = predictor
        self.forecaster = nn_forecasters['forecaster']
        self.nbeats = nn_forecasters['nbeats']
        self.xyz7_model = nn_forecasters['xyz7_model']
        self.features = features
        self.target_scaler = target_scaler
    
        self.num_features = data.shape[-1]
        self.time_window = data.shape[-2]
        self.input_dim = (self.time_window, self.num_features)
        self.forecast_window = self.forecaster.output_shape[-2]
        
        temp_train = data.reshape(-1,self.num_features)
        self.global_mean, self.global_std = [],[]
        for i in range(self.num_features):
            self.global_mean.append(temp_train[:,i].mean())
            self.global_std.append(temp_train[:,i].std())
            
        self.min_target = target.min()
        self.max_target = target.max()
        
    def _lime_predict(self, instance):
        t_instance = np.array([instance]).reshape((len(instance), self.time_window, self.num_features))
        a = self.predictor.predict(t_instance)
        a = np.array([i[0] for i in a]) 
        return a

    def load_instance(self, instance):
        
        self.instance = instance
        model = Ridge(alpha=0.0001,fit_intercept=True,random_state=0)
        
        # Lionets weights
        lionet_weights, real_prediction, local_prediction = self.lionet.explain_instance(self.instance,200,model)
        
        # Lime weights
        explanation, _, _, _ = self.lime.explain_instance(self.instance.flatten(), predict_fn = self._lime_predict, num_features=700)
        weights = OrderedDict(explanation.as_list())
        lime_w = dict(sorted(zip(list([int(wk) for wk in weights.keys()]), list(weights.values()))))
        lime_weights = np.array([lime_w[o] for o in lime_w.keys()])
        
        # iPCA
        pca_weights = {}
        for method in self.ipca.keys():
            [timestep_weights, feature_weights], _ = self.ipca[method].find_importance(self.instance,300,model)
            timestep_weights = timestep_weights.flatten()
            pca_weights[method] = [timestep_weights, feature_weights]
            
        
        self.weights_dict = {'LioNets':{False:[lionet_weights],True:pca_weights['LioNets']},
                             'Lime':{False:[lime_weights],True:pca_weights['Lime']}}
        
        ipca_options = [False,True]
        
        # Original stats
        self.original_instance_statistics = {'LioNets':{},'Lime':{}}
        for method,enable_ipca in product(self.weights_dict.keys(),ipca_options):
            self.original_instance_statistics[method][enable_ipca] = self.moded_instance_statistics(self.instance,method,enable_ipca)
            
        # Recommend modifications
        self.recommendation = {'LioNets':{},'Lime':{}}
        for method,enable_ipca in product(self.weights_dict.keys(),ipca_options):
            self.recommendation[method][enable_ipca] = self.recommend_modifications(method,enable_ipca)
        
        self.seeFtr = 1
        self.original_preds, self.original_ftr_all, self.original_ftr_stats = [],[],[]
        self.mod_preds, self.mod_ftr_all, self.mod_ftr_stats = [],[],[]
        self.expected_preds, self.expected_ftr_all, self.expected_ftr_stats = [],[],[]
        self.load_UI()
    
    def modify(self, weights, ftr, mod, mod_range, uni_mod_val=0, uni_wght_sign=1, select_target=0, xyz7_tm=0, forecast_option=6):
    
        start, end = mod_range[0], mod_range[1]

        mod_instance = self.instance.copy()
        local_mean = self.instance[start-1:end,ftr].mean()

        # ---MODS---        
        if mod == 1: # Uniform
            for i in range(start-1, end):
                if weights.reshape(self.input_dim)[i,ftr] > 0 and uni_wght_sign > 0:
                    mod_instance[i,ftr] = mod_instance[i,ftr] + uni_mod_val
                if weights.reshape(self.input_dim)[i,ftr] < 0 and uni_wght_sign < 0:
                    mod_instance[i,ftr] = mod_instance[i,ftr] + uni_mod_val    
        elif mod == 2: # Local Mean
            mod_instance[start-1:end, ftr] = local_mean    
        elif mod == 3: # Global Mean 
            mod_instance[start-1:end, ftr] = self.global_mean[ftr]   
        elif mod == 4: # Zeros
            mod_instance[start-1:end, ftr] = 0    
        elif mod == 5: # Gaussian Noise
            for i in range(start-1, end):
                np.random.seed(2000+i)
                gaussian_noise = np.random.normal(self.global_mean[ftr], self.global_std[ftr], 1)/10
                mod_instance[i,ftr] += gaussian_noise[0]
            np.clip(mod_instance, 0, 1, out=mod_instance)    
        elif mod == 6: # Neural Forecaster
            prediction = self.forecaster.predict(np.expand_dims(mod_instance,axis=0))
            prediction = prediction.squeeze()
            mod_instance = np.append(mod_instance,prediction,axis=0)
            mod_instance = mod_instance[self.forecast_window:]
        elif mod == 7: # Static Forecaster
            for i in range(mod_instance.shape[1]):
                dif = mod_instance[-1,i] - mod_instance[-(self.forecast_window+1):-1,i]
                temp = np.flip(dif) + mod_instance[-1,i]
                mod_instance[:,i] = np.append(mod_instance[self.forecast_window:,i],temp)  
                np.clip(mod_instance[:,i], 0, 1, out=mod_instance[:,i])
        elif mod == 8: # NBeats Forecaster
            prediction = self.nbeats.predict(np.expand_dims(mod_instance,axis=0))
            prediction = prediction.squeeze()
            mod_instance = np.append(mod_instance,prediction,axis=0)
            mod_instance = mod_instance[self.forecast_window:]
        elif mod == 9: # XYZ7 Forecaster
            start = xyz7_tm*self.forecast_window
            end = self.time_window-self.forecast_window+start
            mod_instance = mod_instance[start:end]
            prediction = self.xyz7_model.predict([np.expand_dims(mod_instance,axis=0), 
                                                  np.array(self.target_scaler.transform([[select_target]]))])
            prediction = prediction.squeeze()
            mod_instance = np.append(mod_instance,prediction,axis=0)

        return mod_instance

    def moded_instance_statistics(self, temp_instance, iml_method, enable_ipca):
        
        model = Ridge(alpha=0.0001,fit_intercept=True,random_state=0)
        
        if enable_ipca:
            real_prediction = self.predictor.predict(np.expand_dims(temp_instance,axis=0)).squeeze()
            [timestep_weights, feature_weights], local_prediction  = self.ipca[iml_method].find_importance(temp_instance,300,model)
            weights = timestep_weights.flatten()
        elif iml_method == 'LioNets':
            weights, real_prediction, local_prediction = self.lionet.explain_instance(temp_instance,200,model)
        else:
            real_prediction = self.predictor.predict(np.expand_dims(temp_instance,axis=0)).squeeze()
            explanation, _, _, _ = self.lime.explain_instance(temp_instance.flatten(), predict_fn = self._lime_predict, num_features=700)
            local_prediction = explanation.local_pred[0]
            weights = OrderedDict(explanation.as_list())
            lime_w = dict(sorted(zip(list([int(wk) for wk in weights.keys()]), list(weights.values()))))
            weights = np.array([lime_w[o] for o in lime_w.keys()])

        features_all = {}
        count = 0
        for j in range(self.time_window):
            count2 = 0
            for i in self.features:
                features_all.setdefault(i,[]).append([j, weights[count+count2], temp_instance[j][count2],
                                                     weights[count+count2]*temp_instance[j][count2]])
                count2 = count2 + 1
            count = count + self.num_features

        if enable_ipca: 
            ftr_stats = [feature_weights]
        else:
            features_std, features_mean, features_max, features_min = [],[],[],[]
            for i in features_all:
                naa = np.array(features_all[i])[:,3]
                features_std.append(naa.std())
                features_mean.append(naa.mean())
                features_max.append(naa.max())
                features_min.append(naa.min())
            ftr_stats = [features_mean,features_std,features_min,features_max]

        return [real_prediction, local_prediction], features_all, ftr_stats
    
    def recommend_modifications(self, iml_method, enable_ipca):
        
        _, _, original_ftr_stats =  self.original_instance_statistics[iml_method][enable_ipca]
        ftr_importance  =  original_ftr_stats[0] #The mean of weights per feature
        
        indexed = list(enumerate(ftr_importance))
        indexed.sort(key=lambda tup: tup[1])
        cls0_ftr = list([i for i, v in indexed[:2]])
        cls1_ftr = list(reversed([i for i, v in indexed[-2:]]))
        #print("Class 0 important features:",features[cls0_ftr[0]], features[cls0_ftr[1]])
        #print("Class 1 important features:",features[cls1_ftr[0]], features[cls1_ftr[1]])

        mods = ['Original', 'Uniform', 'Mean(Local)', 'Mean(Global)', 'Zeros', 'Noise'] 
#         'Forecast (Neural)', 'Forecast (Static)', 'Forecast (N-Beats)']
        wghts = ['Negative Weights', 'Positive Weights']

        cls0_mod_results = []
        cls1_mod_results = []
        unif_tests = [0.1, 0.5, -0.1, -0.5]
              
        weights = self.weights_dict[iml_method][enable_ipca][0]
        for ftr in cls0_ftr:
            temp = []
            for v,w in zip(unif_tests,np.sign(unif_tests)):
                mod_inst = self.modify(weights, ftr, 1, (1,self.time_window), v, w)
                mod_preds = self.predictor.predict(np.array([mod_inst,mod_inst]))[0]
                temp.append((mod_preds[0],ftr,1,v,w))
            for mod in range(2,len(mods)):
                mod_inst = self.modify(weights, ftr, mod, (1,self.time_window))
                mod_preds = self.predictor.predict(np.array([mod_inst,mod_inst]))[0]
                temp.append((mod_preds[0],ftr,mod))
            cls0_mod_results.append(max(temp))

        for ftr in cls1_ftr:
            temp = []
            for v,w in zip(unif_tests,-np.sign(unif_tests)):
                mod_inst = self.modify(weights, ftr, 1, (1,self.time_window), v, w)
                mod_preds = self.predictor.predict(np.array([mod_inst,mod_inst]))[0]
                temp.append((mod_preds[0],ftr,1,v,w))
            for mod in range(2,len(mods)):
                mod_inst = self.modify(weights, ftr, mod, (1,self.time_window))
                mod_preds = self.predictor.predict(np.array([mod_inst,mod_inst]))[0]
                temp.append((mod_preds[0],ftr,mod))
            cls1_mod_results.append(min(temp))

        recommendation = "\t\t\t\t\t\t<<< Recommendations >>>\n\n"
        for e0,rec in enumerate(cls0_mod_results):
            if rec[2]==1:
                recommendation += str(e0+1)+") Try the Uniform modification on feature "+str(self.features[rec[1]])+\
                " with Value: "+str(rec[3])+" on the "+str(wghts[int((1+rec[4])/2)])+" to increase the target value.\n"
            else:
                recommendation += str(e0+1)+") Try the "+str(mods[rec[2]])+" modification on feature "+str(self.features[rec[1]])+ \
                " to increase the target value.\n"

        for e1,rec in enumerate(cls1_mod_results):
            if rec[2]==1:
                recommendation += str(e1+e0+2)+") Try the Uniform modification on feature "+str(self.features[rec[1]])+\
                " with Value: "+str(rec[3])+" on the "+str(wghts[int((1+rec[4])/2)])+" to decrease the target value.\n"
            else:
                recommendation += str(e1+e0+2)+") Try the "+str(mods[rec[2]])+" modification on feature "+str(self.features[rec[1]])+ \
                " to decrease the target value.\n"

        return recommendation
        
    def plot_feature(self, ftr_i, mod_ftr_i, mod, rng_sldr, uni_sldr, rd_btn_uni, select_target, rd_btn_xyz7, forecast_optns, iml_method, enable_ipca):

        # Recommend modifications
        print(self.recommendation[iml_method][enable_ipca])

        # Disable/Enable UI elements
        if mod>=2 and mod<=5:
            self.mod_settings.children = ([self.opt2_settings])
            self.modify_ftr_i.disabled =  False
        elif mod==1:
            self.mod_settings.children = ([self.opt3_settings])
            self.modify_ftr_i.disabled =  False
        elif mod==9:
            self.mod_settings.children = ([self.opt4_settings])
            self.forecast.disabled = False if rd_btn_xyz7 else True
            self.modify_ftr_i.disabled =  True
        else: 
            self.mod_settings.children = ([self.opt1_settings])
            self.modify_ftr_i.disabled =  True

        # If a UI element has been changed other than the Feature View proceed to the modification
        if self.seeFtr == ftr_i:
            if mod==0: 
                self.mod_preds, self.mod_ftr_all, self.mod_ftr_stats = self.original_instance_statistics[iml_method][enable_ipca]
                self.original_preds, self.original_ftr_all, self.original_ftr_stats = self.original_instance_statistics[iml_method][enable_ipca]
            else:
                inst_mod = self.modify(self.weights_dict[iml_method][enable_ipca][0], mod_ftr_i-1, mod, rng_sldr, uni_sldr, rd_btn_uni,
                                  select_target, rd_btn_xyz7, forecast_optns)
                self.mod_preds, self.mod_ftr_all, self.mod_ftr_stats = self.moded_instance_statistics(inst_mod, iml_method,enable_ipca)
                self.original_preds, self.original_ftr_all, self.original_ftr_stats = self.original_instance_statistics[iml_method][enable_ipca]
            if mod==9 and rd_btn_xyz7:
                inst_mod = self.modify(self.weights_dict[iml_method][enable_ipca][0], mod_ftr_i-1, forecast_optns, rng_sldr)
                self.expected_preds, self.expected_ftr_all, self.expected_ftr_stats = \
                self.moded_instance_statistics(inst_mod, iml_method,enable_ipca)              
        else:
            self.seeFtr = ftr_i

        # Print the predictions of Target for the original and modified instance
        print("ORIGINAL -> Real prediction: " + str(self.target_scaler.inverse_transform([[self.original_preds[0]]]).squeeze())[:7] + \
              ", Local prediction: " + str(self.target_scaler.inverse_transform([[self.original_preds[1]]]).squeeze())[:7])
        if mod==9 and rd_btn_xyz7:
            print("EXPECTED -> Real prediction: " + str(self.target_scaler.inverse_transform([[self.expected_preds[0]]]).squeeze())[:7] + \
                  ", Local prediction: " + str(self.target_scaler.inverse_transform([[self.expected_preds[1]]]).squeeze())[:7])
        print("  MOD    -> Real prediction: " + str(self.target_scaler.inverse_transform([[self.mod_preds[0]]]).squeeze())[:7] + \
              ", Local prediction: " + str(self.target_scaler.inverse_transform([[self.mod_preds[1]]]).squeeze())[:7])
        
        
        # Plotting the figures 
        to_vis = self.features
        x = np.arange(len(to_vis))
        if mod==9 and rd_btn_xyz7:
            width = 0.25
            align='edge'
        else:
            width = 0.35
            align='edge'
            
        if enable_ipca:
            fig, axs  = plt.subplots(1,3,figsize=(18, 4))
            axs[1].bar(x-width, self.original_ftr_stats[0], width=width, tick_label=to_vis, align=align, color='C0')
            axs[1].bar(x, self.mod_ftr_stats[0], width=width, tick_label=to_vis, align=align, color='C1')
            axs[1].set_title('Feature Importance ')
            axs[1].legend(('Οriginal','Modded'))
            axs[1].set_xticklabels(to_vis, rotation=45)
            if mod==9 and rd_btn_xyz7:
                axs[1].bar(x+width, self.expected_ftr_stats[0], width=width, tick_label=to_vis, align=align, color='C2')
                axs[1].legend(('Οriginal','Modded','Expected'))
            axs[0].axis('off')
            axs[2].axis('off')
            plt.show()
        else:   
            fig, axs = plt.subplots(1, 3, figsize=(18, 4), dpi=200)
            axs[0].bar(x-width, self.original_ftr_stats[0], width=width, tick_label=to_vis, align=align, color='C0')
            axs[0].bar(x, self.mod_ftr_stats[0], width=width, tick_label=to_vis, align=align, color='C1')
            axs[0].set_xticklabels(to_vis, rotation=45)
            axs[0].set_title('Mean')
            axs[0].legend(('Οriginal','Modded'))
            axs[1].bar(x-width, self.original_ftr_stats[1], width=width, tick_label=to_vis, align=align, color='C0')
            axs[1].bar(x, self.mod_ftr_stats[1], width=width, tick_label=to_vis, align=align, color='C1')
            axs[1].set_xticklabels(to_vis, rotation=45)
            axs[1].set_title('STD')
            axs[2].bar(x-width, self.original_ftr_stats[2], width=width, tick_label=to_vis, align=align, color='C0',)
            axs[2].bar(x-width, self.original_ftr_stats[3], width=width, tick_label=to_vis, align=align, color='C0')
            axs[2].bar(x, self.mod_ftr_stats[2], width=width, tick_label=to_vis, align=align, color='C1')
            axs[2].bar(x, self.mod_ftr_stats[3], width=width, tick_label=to_vis, align=align, color='C1')
            axs[2].set_xticklabels(to_vis, rotation=45)
            axs[2].set_title('Max and Min')
            if mod==9 and rd_btn_xyz7:
                axs[0].bar(x+width, self.expected_ftr_stats[0], width=width, tick_label=to_vis, align=align, color='C2')
                axs[1].bar(x+width, self.expected_ftr_stats[1], width=width, tick_label=to_vis, align=align, color='C2')
                axs[2].bar(x+width, self.expected_ftr_stats[2], width=width, tick_label=to_vis, align=align, color='C2')
                axs[2].bar(x+width, self.expected_ftr_stats[3], width=width, tick_label=to_vis, align=align, color='C2')
                axs[0].legend(('Οriginal','Modded','Expected'))
            plt.show()
            
        
        main_ftr_all = self.expected_ftr_all if mod==9 and rd_btn_xyz7 else self.original_ftr_all
        TIMESTEPS = np.arange(self.instance.shape[0])        
        plt.figure(figsize=(16, 4), dpi=200, facecolor='w', edgecolor='k')
        plt.subplot(131)
        plt.plot(TIMESTEPS,np.array(main_ftr_all[self.features[ftr_i-1]])[:,1],color='grey',linestyle = ':')
        plt.plot(TIMESTEPS,np.array(self.mod_ftr_all[self.features[ftr_i-1]])[:,1],color='tab:blue')
        plt.hlines(y=np.array(self.mod_ftr_all[self.features[ftr_i-1]])[:,1].mean(), xmin=0, xmax=self.time_window, label='mean')
        plt.title(str("Feature\'s " + self.features[ftr_i-1] + " influence"))
        plt.subplot(132)
        plt.plot(TIMESTEPS,np.array(main_ftr_all[self.features[ftr_i-1]])[:,2],color='grey',linestyle = ':')
        plt.plot(TIMESTEPS,np.array(self.mod_ftr_all[self.features[ftr_i-1]])[:,2],color='g')
        plt.hlines(y=np.array(self.mod_ftr_all[self.features[ftr_i-1]])[:,2].mean(), xmin=0, xmax=self.time_window, label='mean')
        plt.title(str("Feature\'s " + self.features[ftr_i-1] + " value"))
        plt.subplot(133)
        plt.plot(TIMESTEPS,np.array(main_ftr_all[self.features[ftr_i-1]])[:,3],color='grey',linestyle = ':')
        plt.plot(TIMESTEPS,np.array(self.mod_ftr_all[self.features[ftr_i-1]])[:,3],color='r')
        plt.hlines(y=np.array(self.mod_ftr_all[self.features[ftr_i-1]])[:,3].mean(), xmin=0, xmax=self.time_window, label='mean')
        plt.title(str("Feature\'s " + self.features[ftr_i-1] + " influence * value"))
        plt.show()

        
    def load_UI(self):
        '''Setting up the interactive visualization tool'''
        
        # UI elements
        range_slider = IntRangeSlider(value=[1,self.time_window], min=1, max=self.time_window, description="Range: ", continuous_update = False)
        view_ftr_i = IntSlider(min=1, max=self.num_features, default_value=2, description="View Feature: ", continuous_update = False)
        self.modify_ftr_i = IntSlider(min=1, max=self.num_features, default_value=2, description="Mod Feature: ", continuous_update = False)
        uniform_slider = FloatSlider(value=0, min=-1, max=1, step=0.05, description='Value:', continuous_update = False)
        radio_button_uni = RadioButtons(options=[('Positive Weights', 1), ('Negative Weights', -1)], description='Affect:')
        select_target = BoundedFloatText(value=(self.min_target+self.max_target)/2, min=self.min_target, max=self.max_target, layout={'width': '150px'})
        radio_button_xyz7 = RadioButtons(options=[('Present ('+str(self.forecast_window)+'-last values)', 0), ('Future ('+str(self.forecast_window)+'-next values)', 1)], description='Affect:')
        enable_iPCA = Checkbox(value=False, description='Enable iPCA')
        iml_method = ToggleButtons(options=['LioNets','Lime'])
        self.forecast = Dropdown(options=[('Neural', 6), ('Static', 7),('N-Beats', 8)], description="Forecast: ")
        mod = Dropdown(options=[('Original', 0), ('Uniform', 1), ('Mean (Local)', 2), ('Mean (Global)', 3), ('Zeros', 4), ('Noise', 5),
                                ('Forecast (Neural)', 6), ('Forecast (Static)', 7), ('Forecast (N-Beats)', 8), ('Forecast (XYZ7)', 9)],
                                description="Mods: ")
        jsdlink((self.modify_ftr_i, 'value'), (view_ftr_i, 'value'))

        # UI layout
        interpretable_settings = HBox([Label('Interpretation method:'), iml_method, enable_iPCA])
        enable_iPCA.layout.margin = '0 0 0 -50px'
        interpretable_settings.layout.margin = '20px 0 20px 0'
        standard_settings = VBox([self.modify_ftr_i,view_ftr_i])
        xyz7_settings = VBox([HBox([Label('Desired Target:'), select_target]), radio_button_xyz7])
        xyz7_settings.layout.margin = '0 0 0 30px'
        self.opt1_settings = VBox([mod])
        self.opt2_settings = VBox([mod,range_slider])
        self.opt3_settings = HBox([VBox([mod, range_slider]),VBox([uniform_slider, radio_button_uni])])
        self.opt4_settings = HBox([VBox([mod, self.forecast]),xyz7_settings])
        self.mod_settings = VBox([])
        ui = VBox([interpretable_settings, HBox([standard_settings,self.mod_settings])])

        # Starting the interactive tool
        inter = interactive_output(self.plot_feature, {'ftr_i':view_ftr_i, 'mod_ftr_i':self.modify_ftr_i,'mod':mod,
                                'rng_sldr':range_slider, 'uni_sldr':uniform_slider, 'rd_btn_uni':radio_button_uni, 'select_target':select_target, 
                                'rd_btn_xyz7':radio_button_xyz7, 'forecast_optns':self.forecast, 'iml_method':iml_method, 'enable_ipca':enable_iPCA})
        display(ui,inter)
