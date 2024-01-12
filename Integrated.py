import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from AI.Github.CNNModel2 import CNN2
from AI.Github.PersonalTry import Persona


# ... (your existing code)

class IntegratedModel:
    def __init__(self, num_hidden_states, cnn_model):
        self.num_hidden_states = num_hidden_states
        self.hmm_model = hmm.MultinomialHMM(n_components=num_hidden_states)
        self.cnn_model = cnn_model

    def train(self, input_data, output_data, batch_size, epochs, learning_rate):
        # Train the HMM using input_data (historical schedules)
        encoded_output_data = self.encode_output_data(input_data)
        print(encoded_output_data.dtype, encoded_output_data.shape, encoded_output_data.ndim, "\n", "Encoded Output Data.dtype, Encoded Output Data.shape, Encoded Output Data.ndim")

        self.hmm_model.fit(encoded_output_data)

        # Use the trained HMM to generate sequences of hidden states
        hidden_states = self.hmm_model.predict(encoded_output_data)

        # Use hidden states as inputs to the CNN
        cnn_input = self.transform_hidden_states(hidden_states)

        # Train the CNN using the integrated input
        self.cnn_model.train(cnn_input, output_data, batch_size, epochs, learning_rate)

    def encode_output_data(self, output_data):
        # Use one-hot encoding to convert output_data to the format expected by hmmlearn
        encoder = OneHotEncoder(categories='auto', sparse=False)
        encoded_output_data = encoder.fit_transform(output_data.flatten().reshape(-1, 1))

        # Cast the encoded output data to integers
        encoded_output_data = encoded_output_data.astype(int)

        return encoded_output_data

    def transform_hidden_states(self, hidden_states):
        # Simple transformation - convert hidden states to one-hot encoding
        encoder = OneHotEncoder(categories='auto', sparse=False)
        transformed_data = encoder.fit_transform(hidden_states.reshape(-1, 1))
        return transformed_data.reshape(hidden_states.shape[0], -1)

    def predict(self, input_data):
        # Reshape the input data to match the expected shape
        reshaped_input = input_data.reshape(input_data.shape[0], -1, input_data.shape[-1])

        # Use the HMM to predict hidden states
        hidden_states = self.hmm_model.predict(reshaped_input)

        # Use hidden states as inputs to the CNN for predictions
        cnn_input = self.transform_hidden_states(hidden_states)

        # Make predictions using the CNN
        predictions = self.cnn_model.model.predict(cnn_input)

        return predictions


# Example usage of the integrated model
cnn_model = CNN2()
personas = cnn_model.load_schedules("output2.xlsx", num_personas=4)

stacked_schedules = Persona.stack_schedules(personas[0].num_weeks, *personas)
print(stacked_schedules)
stacked_schedules = np.array(stacked_schedules)
print(stacked_schedules.shape, stacked_schedules.dtype, stacked_schedules.ndim, "\n", "Stacked Schedules.shape, Stacked Schedules.dtype, Stacked Schedules.ndim")
overlap_matrix = Persona.check_overlap(stacked_schedules)
stacked_schedules_transform = Persona.transform_matrix(stacked_schedules)
overlap_matrix_transform = Persona.transform_matrix(overlap_matrix)

#
# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(stacked_schedules_transform, overlap_matrix_transform,
                                                    test_size=0.2, random_state=42)
learning_rate = 0.001
cnn_model.train(X_train, y_train, batch_size=32, epochs=2250, learning_rate=learning_rate, validation_data=(X_test, y_test))
num_hidden_states = 3  # You can adjust this based on your problem
integrated_model = IntegratedModel(num_hidden_states, cnn_model)
learning_rate = 0.001  # You can adjust this based on your problem

# Training the integrated model
integrated_model.train(X_train, y_train, batch_size=32, epochs=450, learning_rate=learning_rate)

# Making predictions using the integrated model
predictions = integrated_model.predict(X_test)


