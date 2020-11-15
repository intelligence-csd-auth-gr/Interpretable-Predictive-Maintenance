from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def find_importance(instance, lionet, n_neighbours, scaling='global'):

    # Create instance neighbourhood
    neighbourhood, predictions = lionet.give_me_the_neighbourhood(instance,n_neighbours)
    neighbourhood = neighbourhood.reshape(-1,50,14)

    # Perform PCA per sensor 
    reduced_data = []
    pca_list = []
    for sens in range(14):
        pca = PCA(n_components=1)
        reduced_data.append(pca.fit_transform(neighbourhood[:,:,sens]))
        pca_list.append(pca)

    reduced_data = np.squeeze(np.array(reduced_data).T)

    # Scaling data to [0.1,1,1]
    if scaling == 'global': 
        t_min, t_max = np.min(reduced_data), np.max(reduced_data)
        scaled_data = reduced_data/(t_max-t_min)-t_min/(t_max-t_min)+0.1
    elif scaling == 'individual':
        scaler = MinMaxScaler(feature_range=(0.1,1.1))
        scaled_data = scaler.fit_transform(reduced_data)

    # Train a Ridge Regression model with the transformed data
    rr = Ridge(random_state=0)
    rr.fit(scaled_data,predictions)

    # Extract the importance(weight) of each sensor for the given instance (last element of the neighbourhood)
    sensor_weights = scaled_data[-1] * rr.coef_
    sensor_weights = np.array(sensor_weights)
    
    # Extract the importance(weight) of each timestep per sensor for the given instance (last element of the neighbourhood)
    timestep_weights = []
    for e,pca in enumerate(pca_list):
        timestep_weights.append(pca.components_[0]*instance[:,e])
    timestep_weights = np.array(timestep_weights).T
    
    return sensor_weights, timestep_weights