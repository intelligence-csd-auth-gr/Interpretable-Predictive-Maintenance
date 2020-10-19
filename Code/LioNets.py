import matplotlib.pyplot as plt
import random
import seaborn as sns
import numpy as np
from math import sqrt, exp, log
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, SGDRegressor
import pandas as pd

class LioNet:
    """Class for interpreting a neural network locally"""
    def __init__(self, predictor, decoder, encoder, train_data, target_scaler=None, feature_names=None, decoder_lower_threshold=0):
        """Init function
        Args:
            model: The trained predictor model
            autoencoder: The trained autoencoder
            decoder: The second half of the autoencoder
            encoder: The first half of the autoencoder
            train_data: The training data are necessary to compute their encoded feature's statistics
            feature_names: The selected features. The above networks have been trained with these.
            decoder_lower_threshold: Only for tfidf decoders!
        """
        self.predictor = predictor
        self.decoder = decoder
        self.encoder = encoder
        self.train_data = train_data
        self.target_scaler = target_scaler
        self.feature_names = feature_names
        self.decoder_lower_threshold = decoder_lower_threshold
        self.features_statistics = {}
        self.encoded_training_data = encoder.predict(train_data)
        self._extract_feature_statistics()

    def explain_instance(self, new_instance, max_neighbours=None, model=None, random_state=0):
        """Generates the explanation for an instance
        Args:
            new_instance: The instance to explain
            max_neighbours:
            model:
            random_state: 
        Return:
            weights: 
            instance_prediction: 
            local_prediction:
        """

        neighbourhood, predictions, distances = self._get_decoded_neighbourhood(new_instance, max_neighbours, random_state)

        instance_prediction = predictions[-1]

        #if not 1D representation reshape
        if len(new_instance.shape) == 2: #only for 1 or 2 dimension
            one_dimension_size = new_instance.shape[0] * new_instance.shape[1]
            neighbourhood = neighbourhood.reshape((len(neighbourhood),one_dimension_size))

        #train linear model
        linear_model = self._fit_linear_model(neighbourhood, predictions, distances, model)
        weights = linear_model.coef_

        local_prediction = linear_model.predict([neighbourhood[-1]])[0]
        return weights, instance_prediction, local_prediction

    def _get_decoded_neighbourhood(self, instance, max_neighbours=None, random_state=0):
        """Returns
        """
        encoded_instance = self.encoder.predict(np.array([instance]))[0]
        #Add an try catch if encoded feature has not 1D representation
        if max_neighbours is None:
            max_neighbours = len(encoded_instance)*4
            if max_neighbours < 1000:
                max_neighbours = 1000
        encoded_neighbourhood = self._neighbourhood_generation(encoded_instance, max_neighbours, random_state)
        decoded_neighbourhood = self.decoder.predict(np.array(encoded_neighbourhood))
        temp_neighbourhood = [neighbour for neighbour in decoded_neighbourhood]
        temp_neighbourhood.append(instance)
        neighbourhood = np.array(temp_neighbourhood)

        if self.decoder_lower_threshold != 0: #TFIDF vectors:
            for neighbour in range(len(neighbourhood)):
                for feature in range(len(neighbour)):
                    if neighbourhood[neighbour][feature] < self.decoder_lower_threshold:
                        neighbourhood[neighbour][feature]=0

        temp_distances = euclidean_distances([encoded_instance],self.encoder.predict(neighbourhood))[0]
        distances = []
        dimensions = len(encoded_instance)
        if dimensions < 100:
            dimensions = 100
        for distance in temp_distances:
            distances.append(exp(-(distance*log(dimensions)/2))*log(dimensions))

        predictions = self.predictor.predict(neighbourhood)
        if self.target_scaler is not None:
        	predictions = self.target_scaler.inverse_transform(predictions)
        predictions = [prediction[0] for prediction in predictions]

        return neighbourhood, predictions, distances



    def _extract_feature_statistics(self):
        encoded_dim = len(self.encoded_training_data[0])
        for feature in range(encoded_dim):
            self.features_statistics[feature] = []
        for feature in range(encoded_dim):
            self.features_statistics[feature].append(
                self.encoded_training_data[:, feature:feature + 1].min())
            self.features_statistics[feature].append(
                self.encoded_training_data[:, feature:feature + 1].max())
            self.features_statistics[feature].append(
                self.encoded_training_data[:, feature:feature + 1].mean())
            self.features_statistics[feature].append(
                self.encoded_training_data[:, feature:feature + 1].std())

    def _neighbourhood_generation(self, encoded_instance, max_neighbours, random_state=0):
        """Generates the neighbourhood of an instance
        Args:
            encoded_instance: The instance to generate neighbours
            max_neighbours: Maximum number of neighbours
            random_state: A seed for stable neighbours generation
        Return:
            neighbours: The generated neighbours
        """
        encoded_dim = len(encoded_instance)
        neighbours = []
        neighbours.append(encoded_instance)
        for feature in range(encoded_dim): #Adding neighbours different by one element
            value = encoded_instance[feature]
            neighbour_copy = encoded_instance.copy()
            
            v1 = self._determine_feature_change(value, feature, random_state=random_state)
            neighbour_copy[feature] = v1
            neighbours.append(neighbour_copy)
            
            v2 = self._determine_feature_change(value, feature, smaller=True, random_state=random_state)
            if v1 != v2:
                neighbour_copy[feature] = v2
                neighbours.append(neighbour_copy)
            
            v3 = self._determine_feature_change(value, feature, bigger=True, random_state=random_state)
            if v3 != v1 and v3 != v2:
                neighbour_copy[feature] = v3
                neighbours.append(neighbour_copy)
        while len(neighbours) < max_neighbours:
            neighbour_copy = encoded_instance.copy()
            np.random.seed(abs(int((neighbour_copy.sum()**2)*100)+len(neighbours))+random_state)
            for f_i in np.random.randint(2, size=encoded_dim).nonzero()[0]:
                value_to_change = neighbour_copy[f_i]
                new_value = self._determine_feature_change(value_to_change, f_i, smaller=True, random_state=random_state)
                neighbour_copy[f_i] = new_value
            neighbours.append(neighbour_copy)
        neighbours = neighbours[:max_neighbours]
        return neighbours

    def _determine_feature_change(self, value, feature, smaller=False, bigger=False, random_state=0):
        """Determines the new value for a specific feature based on gaussian noise
        Args:
            value: Current value of encoded feature
            feature: The feature itself, in order to find its statistics
            smaller: If we want to draw a smaller gaussian noise, and alter a little the original value
            bigger: If we want to draw a bigger gaussian noise, and alter a lot the original value
            random_state: A seed for stable value generation
        Return:
            new_value: The proposed new value
        """
        min_ = self.features_statistics[feature][0]
        max_ = self.features_statistics[feature][1]
        mean_ = self.features_statistics[feature][2]
        std_ = self.features_statistics[feature][3]
        if smaller:
            std_ = std_ / 2
        elif bigger:
            std_ = std_ * 2
        np.random.seed(abs(int((value**2)*100)+int((mean_**2)*100)+int((std_**2)*100)+random_state))
        new_value = np.random.normal(mean_,std_,1)[0] #Gaussian Noise
        if (value + new_value) < self.features_statistics[feature][0]:
            new_value = self.features_statistics[feature][0]
        elif (value + new_value) > self.features_statistics[feature][1]:
            new_value = self.features_statistics[feature][1]
        else:
            new_value = value + new_value
        return new_value

    def _fit_linear_model(self, X,y,distances, model=None):
        """Determines the new value for a specific feature based on gaussian noise
        Args:
            X:
            y:
            distances:
        Return:
            best_model: 
        """
        if model is None:
            best_model = None
            best_score = 10000
            alphas = [0.1, 1, 10, 100, 1000]
            for a in alphas:
                temp_model = SGDRegressor(alpha=a, penalty='elasticnet').fit(X,y,sample_weight=distances)
                temp_performance = abs(mean_squared_error(y,temp_model.predict(X)))
                if best_score > temp_performance:
                    best_score = temp_performance
                    best_model = temp_model
                temp_model = Ridge(alpha=a, fit_intercept=True).fit(X,y,distances)
                temp_performance = abs(mean_squared_error(y,temp_model.predict(X)))
                if best_score > temp_performance:
                    best_score = temp_performance
                    best_model = temp_model
            return best_model
        else:
            return model.fit(X,y,distances)

