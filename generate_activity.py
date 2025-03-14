"""
This module generates synthetic data for activity recommendations research demonstration, simulating
user behaviors and rewards under different contextual conditions.
"""
import numpy as np
import pandas as pd


def generate_synthetic_data(activities, context, days, records_per_day):
    """
    Generate synthetic data for activity recommendations with versatile context handling.

    Args:
    - activities (list): List of activities.
    - context (dict): Dictionary mapping context names to their possible values.
    - days (int): Number of days for which to generate data.
    - records_per_day (int): Number of records to generate per day.

    Returns:
    - DataFrame containing the synthetic data.
    """
    all_data = []

    for day in range(days):
        for _ in range(records_per_day):
            context_values = {context_name: np.random.choice(values) for context_name, values in context.items()}
            
            # Calculate action probabilities and normalize
            probabilities = calculate_action_probabilities(activities, day)
            action = np.random.choice(activities, p=probabilities)
            
            # Simple reward mechanism based on the action and day
            # reward = 1 if (action in ['Gym', 'Walking'] and day < 50) or (action in ['Yoga', 'Reading'] and day >= 50) else 0
            if (action in ['Gym', 'Walking'] and day < 50) or (action in ['Yoga', 'Reading'] and day >= 50):
                # reward = 1
                if action in ['Gym', 'Walking'] and context_values["time_slots"] in ["9-12", "12-15", "15-18"]:
                    reward = 1
                elif action in ['Yoga', 'Reading'] and context_values["time_slots"] in ['18-21', '21-24']:
                    reward = 1
                else:
                    reward = 0 
            else:
                reward = 0 
            
            # Prepare record
            record = context_values
            record.update({'day': day, 'activity': action, 'reward': reward})
            all_data.append(record)

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(all_data)
    
    return df

def calculate_action_probabilities(activities, day):
    """
    Calculate probabilities for each activity based on the day.
    This function is a placeholder for a more sophisticated logic.
    """
    # Simple example logic to generate probabilities
    num_activities = len(activities)
    probabilities = np.ones(num_activities) / num_activities  # Equal probability for simplicity
    return probabilities

    

if __name__ == "__main__":
    # Generate synthetic data
    # Example usage:
    activities = ['Gym', 'Walking', 'Yoga', 'Reading', 'Meditation']
    contexts = {
        "time_slots": ['6-9', '9-12', '12-15', '15-18', '18-21', '21-24'],
        "temperature": ['Low', 'Medium', 'High'],
        "weather": ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
    }
    days = 100
    records_per_day = 10

    df = generate_synthetic_data(activities, contexts, days, records_per_day)
    print(df.head())


