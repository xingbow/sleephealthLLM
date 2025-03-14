"""
This module implements a contextual multi-armed bandit model for personalized activity recommendations.

Our implementation of contextual bandits refers to the original Github repository: https://github.com/david-cortes/contextualbandits/
License: [BSD-2-Clause license](https://github.com/david-cortes/contextualbandits/?tab=BSD-2-Clause-1-ov-file#readme)
"""
import os
import pandas as pd
import numpy as np
from copy import deepcopy

from sklearn.preprocessing import OneHotEncoder
from contextualbandits.online import BootstrappedUCB
from sklearn.linear_model import LogisticRegression, SGDClassifier

import generate_activity as ga
import globalVariable as GV
import utils
import dill as pickle

activities = GV.ACTIVITIES # list of activities

# contexts include time_slots, temperature_categories, weather_conditions
contexts = {
    "time_slots": GV.TIME_CAT,
    "temperature_categories": GV.TEMP_3_CAT,
    "weather_conditions": GV.WEATHER_TYPE
}

class ActivityRecommender:
    """
    A contextual multi-armed bandit model for personalized activity recommendations.
    """
    def __init__(self, activities, context_encoder = OneHotEncoder(), config: dict = {
        "nsamples": 10,
        "percentile": 20
    }):
        """
        Initialize the ActivityRecommender class.

        Args:
            activities (list): List of activities to be recommended.
            context_encoder (OneHotEncoder): Encoder for contextual features.
            config (dict): Configuration for the model.
        """
        self.activities = activities
        self.encoder = context_encoder
        self.base_algorithm = SGDClassifier(loss='log_loss', warm_start=False, random_state=123)
        # self.base_algorithm = LogisticRegression(solver='lbfgs', warm_start=True)
        self.model = BootstrappedUCB(deepcopy(self.base_algorithm), 
                                     len(activities), 
                                     random_state=111,
                                     batch_train=True,
                                     nsamples=config["nsamples"], 
                                     percentile=config["percentile"])
    def add_activity(self, activity: str):
        """
        Add a new activity to the model.

        Args:
            activity (str): The new activity to be added.
        """
        self.activities.append(activity)
        self.model.add_arm()

    def train(self, X, y, rewards, partial_fit: bool = True):
        """
        Train the model on a batch of data.

        Args:
            X (array-like): Contextual features.
            y (array-like): Actions.
            rewards (array-like): Rewards.
            partial_fit (bool): Whether to use partial fit.
        """
        if partial_fit:
            self.model.partial_fit(X, y, rewards)
        else:
            self.model.fit(X, y, rewards)


    def predict(self, context):
        """
        Predict the recommended action for a given context.

        Args:
            context (array-like): The contextual features.

        Returns:
            str: The recommended action.
        """
        context_encoded = self.encoder.transform(context)
        recommended_action_index = self.model.predict(context_encoded)
        recommended_action = self.activities[recommended_action_index[0]]

        return recommended_action

    def predict_topN_above_threshold(self, context, n=2, threshold=.9):
        """
        Predict the top N actions above a given threshold for a given context.

        Args:
            context (array-like): The contextual features.
            n (int): The number of actions to return.
            threshold (float): The threshold for the action scores.

        Returns:
            list: The top N actions above the threshold.
        """
        context_encoded = self.encoder.transform(context)
        action_scores = self.model.decision_function(context_encoded)[0]
        activity_indexs = utils.filter_and_sort_scores(action_scores, threshold)
        if len(activity_indexs) == 0:
            return [self.activities[np.argmax(action_scores)]]
        else:
            return [self.activities[i] for i in activity_indexs[:n]]

if __name__ == "__main__":
    # Generating synthetic data to demonstrate the model
    days = 100
    records_per_day = 10
    df = ga.generate_synthetic_data(activities, contexts, days, records_per_day)
    
    # Encoding the contextual features
    encoder = OneHotEncoder()
    contextual_features = df[list(contexts.keys())].values
    X = encoder.fit_transform(contextual_features)

    # Mapping activities to indices
    activity_to_index = {activity: i for i, activity in enumerate(activities)}
    y = df['activity'].map(activity_to_index).values

    # Rewards
    rewards = df['reward'].values

    # Initialize recommendation model
    model = ActivityRecommender(activities, encoder)
    # model.train(X, y, rewards)
    for i in range(len(y)):
        model.train(X[i,:], np.array([y[i]]), np.array([rewards[i]]))
    
    # # Save the model
    # with open("recommender_model.pkl", "wb") as f:
    #     pickle.dump(model, f)

    # Predicting for a new context
    new_context = [['9-12', 'Mild', 'sunny']]
    recommended_action = model.predict(new_context)

    print(f"Recommended action for {new_context}:  {recommended_action}, model: {model.model.decision_function(encoder.transform(new_context))}")
    # add new activity
    new_activity = 'Swimming'
    model.add_activity(new_activity)
    # model.train(X, y, rewards)
    new_context = [['9-12', 'Mild', 'sunny']]
    recommended_action = model.predict(new_context)
    print(f"Recommended action for {new_context}:  {recommended_action} , model: {model.model.decision_function(encoder.transform(new_context))}")


