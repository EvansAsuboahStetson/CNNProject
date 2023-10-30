import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Reshape

from AI.Github.Persona import Persona

# import openpyxl
from openpyxl import Workbook
from openpyxl import load_workbook
from openpyxl import Workbook
from openpyxl import load_workbook


# Ensure that the Persona class is available in the current file or import it correctly.

class CNN:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(16, 5, 3)))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(240, activation='sigmoid'))  # Adjust the output shape based on your requirements
        model.add(Reshape((16, 5, 3)))
        model.summary()
        return model

    def train(self, input_data, output_data, batch_size, epochs):
        print(input_data.shape)
        print(output_data.shape)
        self.model.compile(optimizer='adam', loss='mse')  # You can change the loss function based on your specific task
        self.model.fit(input_data, output_data, batch_size=batch_size, epochs=epochs)

    @staticmethod
    def load_schedules(file_path):
        persona1 = Persona("Persona1", 9, 17, 60, 52)
        persona2 = Persona("Persona2", 9, 17, 60, 52)
        return Persona.setSchedulesFromLoad(file_path=file_path, persona1=persona1, persona2=persona2)

    @staticmethod
    def load_and_predict_validation(file_path, cnn_model):
        persona1, persona2 = cnn_model.load_schedules(file_path=file_path)
        stacked_schedules = Persona.stack_schedules(persona1, persona2, persona1.num_weeks)
        stacked_schedules = np.array(stacked_schedules)
        new_stacked_schedules_transform = Persona.transform_matrix(stacked_schedules)
        predictions_new = cnn_model.model.predict(new_stacked_schedules_transform)
        return predictions_new

    @staticmethod
    def plot_overlaps(predicted_overlaps, actual_overlaps, week_number):
        reshaped_predictions = predicted_overlaps.reshape((predicted_overlaps.shape[0], 16, 5, 3))

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the predicted overlaps
        axs[0].imshow(reshaped_predictions[0, :, :, 0], cmap='coolwarm', interpolation='nearest')
        axs[0].set_title('Predicted Overlaps')
        axs[0].set_xlabel('Days')
        axs[0].set_ylabel('Time Slots')

        # Annotate the overlapping times for the predicted overlaps
        for i in range(16):
            for j in range(5):
                text = axs[0].text(j, i, reshaped_predictions[0, i, j, 0],
                                   ha="center", va="center", color="w")

        # Reshape the actual overlaps to fit the plot
        reshaped_actual_overlaps = actual_overlaps.reshape(16, 5, 3)

        # Plot the actual overlaps
        axs[1].imshow(reshaped_actual_overlaps[:, :, 0], cmap='coolwarm', interpolation='nearest')
        axs[1].set_title('Actual Overlaps')
        axs[1].set_xlabel('Days')
        axs[1].set_ylabel('Time Slots')

        # Annotate the overlapping times for the actual overlaps
        for i in range(16):
            for j in range(5):
                text = axs[1].text(j, i, reshaped_actual_overlaps[i, j, 0],
                                   ha="center", va="center", color="w")

        # Add the week number as a title
        fig.suptitle(f'Week {week_number}', fontsize=16)

        plt.show()

    def convert_and_evaluate(self, predicted_overlaps, overlap_matrix_transform, week_to_predict):
        threshold_low = 0.3
        threshold_medium = 0.6
        threshold_high = 0.9

        # Convert the predicted values based on the thresholds
        predicted_overlaps_classes = np.zeros_like(predicted_overlaps, dtype=int)
        predicted_overlaps_classes[predicted_overlaps > threshold_high] = 2
        predicted_overlaps_classes[
            np.logical_and(predicted_overlaps > threshold_medium, predicted_overlaps <= threshold_high)] = 1
        predicted_overlaps_classes[predicted_overlaps <= threshold_low] = 0

        # Print the actual overlaps for the week
        actual_overlaps = overlap_matrix_transform[week_to_predict]

        total_samples = np.prod(actual_overlaps.shape)
        correct_predictions = np.sum(predicted_overlaps_classes == actual_overlaps)
        accuracy = correct_predictions / total_samples
        return predicted_overlaps_classes, accuracy
    @staticmethod
    def display_overlaps_info(week_to_predict, persona1, predicted_overlaps_classes, actual_overlapsValidation):
        print("\nDate and Time for Predicted OverlapsValidation:")
        for i in range(16):
            for j in range(5):
                if predicted_overlaps_classes[0, i, j, 0] != 0:
                    print(
                        f"Overlap at Week {week_to_predict}, Day {persona1.days[j]}, Time {persona1.start_hour + i * persona1.interval_minutes / 60:.2f}")

        # Display date and time for the actual overlaps
        print("\nDate and Time for Actual Overlaps:")
        for i in range(16):
            for j in range(5):
                if actual_overlapsValidation[i, j, 0] != 0:
                    print(
                        f"Overlap at Week {week_to_predict}, Day {persona1.days[j]}, Time {persona1.start_hour + i * persona1.interval_minutes / 60:.2f}")

    # Example usage:
    # Assuming you have the necessary variables defined, you can call the function like this:
    # display_overlaps_info(week_to_predict, persona1, predicted_overlaps_classes, actual_overlapsValidation)


cnn_model = CNN()

# Load the schedules
persona1, persona2 = cnn_model.load_schedules("output.xlsx")
print(persona1.num_weeks, "num_weeks")

stacked_schedules = Persona.stack_schedules(persona1, persona2, persona1.num_weeks)

stacked_schedules = np.array(stacked_schedules)

overlap_matrix = Persona.check_overlap(stacked_schedules)
stacked_schedules_transform = Persona.transform_matrix(stacked_schedules)
overlap_matrix_transform = Persona.transform_matrix(overlap_matrix)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(stacked_schedules_transform, overlap_matrix_transform,
                                                    test_size=0.2, random_state=42)

# Train the model on the training data

cnn_model.train(X_train, y_train, batch_size=32, epochs=350)

# Make predictions using the trained model
predictions = cnn_model.model.predict(X_test)

# Print the predictions

week_to_predict = 42

# Reshape the data for prediction
X_to_predict = stacked_schedules_transform[week_to_predict].reshape(1, 16, 5, 3)

# Make predictions for the specified week
predicted_overlaps = cnn_model.model.predict(X_to_predict)

# Print the actual overlaps for the week
actual_overlaps = overlap_matrix_transform[week_to_predict]

predicted_overlaps_classes, accuracy = cnn_model.convert_and_evaluate(
    predicted_overlaps, overlap_matrix_transform, week_to_predict
)

print(f"Accuracy: {accuracy * 100:.2f}%")

cnn_model.plot_overlaps(predicted_overlaps_classes, actual_overlaps, week_to_predict)
file_path = "ValidationOutput.xlsx"

persona1.add_weeks(48)
persona2.add_weeks(48)

persona1, persona2 = Persona.setSchedulesFromLoad(file_path=file_path, persona1=persona1, persona2=persona2)
print(persona1.num_weeks, "num_weeks_after Validation")

stacked_schedulesValidation = Persona.stack_schedules(persona1, persona2, persona1.num_weeks)

stacked_schedulesValidation = np.array(stacked_schedulesValidation)
print(stacked_schedulesValidation.shape, "stacked_schedules.shape")
new_stacked_schedules_transform = Persona.transform_matrix(stacked_schedulesValidation)

week_to_predictValidation = 53  # Week three is at index 2
X_to_predictValidation = new_stacked_schedules_transform[week_to_predictValidation].reshape(1, 16, 5, 3)
predictions_new = cnn_model.model.predict(X_to_predictValidation)

overlap_matrixValidation = Persona.check_overlap(stacked_schedulesValidation)
overlap_matrix_transformValidation = Persona.transform_matrix(overlap_matrixValidation)
actual_overlapsValidation = overlap_matrix_transformValidation[week_to_predictValidation]

# Print the predictions


predicted_overlaps_classesValidation, accuracyValidation = cnn_model.convert_and_evaluate(
    predictions_new, overlap_matrix_transformValidation, week_to_predictValidation
)

print(f"Accuracy: {accuracyValidation * 100:.2f}%")

cnn_model.plot_overlaps(predicted_overlaps_classesValidation, actual_overlapsValidation, week_to_predictValidation)
print(predicted_overlaps_classes, "predicted_overlaps_classesValidation")

cnn_model.display_overlaps_info(week_to_predictValidation, persona1,
                                predicted_overlaps_classesValidation,actual_overlapsValidation)
