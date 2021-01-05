from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np

class iPCA:
    def __init__(self, neighbours_generator=None, scaling='global'):
        """Init function
        Args:
            model: The trained RF model
        Attributes:
            model:
        """
        self.neighbours_generator = neighbours_generator
        self.scaling = scaling
 
    
    def find_importance(self, instance, n_neighbours=300, model=None):

        # Create instance neighbourhood
        neighbourhood, predictions, distances = self.neighbours_generator(instance,n_neighbours)
        neighbourhood = neighbourhood.reshape(-1,50,14)

        # Perform PCA per sensor
        reduced_data = []
        pca_list = []
        for sens in range(14):
            pca = PCA(n_components=1, svd_solver='arpack')
            reduced_data.append(pca.fit_transform(neighbourhood[:,:,sens]))
            pca_list.append(pca)

        reduced_data = np.squeeze(np.array(reduced_data).T)

        # Scaling data to [0.1,1,1]
        if self.scaling == 'global':
            t_min, t_max = np.min(reduced_data), np.max(reduced_data)
            scaled_data = reduced_data/(t_max-t_min)-t_min/(t_max-t_min)+0.1
        elif self.scaling == 'local':
            scaler = MinMaxScaler(feature_range=(0.1,0.9))
            scaled_data = scaler.fit_transform(reduced_data)

        # Train a Ridge Regression model with the transformed data
        rr = self._fit_linear_model(scaled_data, predictions, distances, model)
#         rr = Ridge(alpha=0.001,random_state=0)
#         rr.fit(scaled_data,predictions)
        
        # Obtaining the local prediction for the instance
        local_prediction = rr.predict(scaled_data[-1].reshape(1,-1)).squeeze()

        # Extract the importance(weight) of each sensor for the given instance (last element of the neighbourhood)
        sensor_weights = rr.coef_
        sensor_weights = np.array(sensor_weights)
        
        # Extract the importance(weight) of each timestep per sensor for the given instance (last element of the neighbourhood)
        timestep_weights = []
        for e,pca in enumerate(pca_list):
            timestep_weights.append(pca.components_[0])
        timestep_weights = np.array(timestep_weights).T
        
        return [timestep_weights, sensor_weights], local_prediction
    
    
    def _fit_linear_model(self, X, y, distances, model):
        
        if model is None:
            best_model = None
            best_score = 10000
            alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            for a in alphas:
                temp_model = Ridge(alpha=a, fit_intercept=True).fit(X,y,distances)
                temp_performance = abs(mean_squared_error(y,temp_model.predict(X)))
                if best_score > temp_performance:
                    best_score = temp_performance
                    best_model = temp_model
                    self.alpha = a
            return best_model
        else:
            return model.fit(X,y,distances)
