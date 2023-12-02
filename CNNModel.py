import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Reshape, Conv2DTranspose
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from AI.Github.Persona import Persona

class CNN:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(16, 5, 3)))
        print(f"Shape after Conv2D: {model.output_shape}")

        # Add MaxPooling2D layer
        model.add(MaxPooling2D((2, 1), strides=(2, 1), padding='same'))
        print(f"Shape after MaxPooling2D: {model.output_shape}")

        # Add another Conv2D layer
        model.add(Conv2D(64, (3, 3), activation='sigmoid', padding='same'))
        print(f"Shape after Conv2D: {model.output_shape}")

        # Add another MaxPooling2D layer
        model.add(MaxPooling2D((2, 1), strides=(2, 1), padding='same'))
        print(f"Shape after MaxPooling2D: {model.output_shape}")

        # Add another Conv2D layer
        model.add(Conv2D(128, (3, 3), activation='sigmoid', padding='same'))
        print(f"Shape after Conv2D: {model.output_shape}")

        # Add another MaxPooling2D layer
        model.add(MaxPooling2D((2, 1), strides=(2, 1), padding='same'))
        print(f"Shape after MaxPooling2D: {model.output_shape}")

        # Add Conv2DTranspose layer
        model.add(Conv2DTranspose(64, (3, 3), strides=(2, 1), padding='same'))
        print(f"Shape after Conv2DTranspose: {model.output_shape}")

        # Add another Conv2DTranspose layer
        model.add(Conv2DTranspose(32, (3, 3), strides=(2, 1), padding='same'))
        print(f"Shape after Conv2DTranspose: {model.output_shape}")

        # Add the final Conv2DTranspose layer
        model.add(Conv2DTranspose(3, (3, 3), strides=(2, 1), activation='sigmoid', padding='same'))
        print(f"Final output shape: {model.output_shape}") # Final output layer

        model.summary()
        return model

    def train(self, input_data, output_data, batch_size, epochs, learning_rate):
        print(input_data.shape)
        print(output_data.shape)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='logcosh',
                           metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                                    tf.keras.metrics.AUC()])

        # Check if the trained model file exists
        model_filename = "trained_model.h5"
        if os.path.exists(model_filename):
            # Load the trained model
            self.model = load_model(model_filename)
        else:
            # Train the model
            history = self.model.fit(input_data, output_data, batch_size=batch_size, epochs=epochs)

            print("Accuracy:", history.history['accuracy'])
            print("Precision:", history.history['precision'])
            print("Recall:", history.history['recall'])
            print("AUC:", history.history['auc'])

            # Save the trained model
            self.model.save(model_filename)

            # Plot the loss curve
            self.plot_loss_curve(history)

    def plot_loss_curve(self, history):
        # Retrieve the loss values from the history object
        loss = history.history['loss']
        epochs = range(1, len(loss) + 1)

        # Plot the loss curve
        plt.plot(epochs, loss, 'g', label='Training loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

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
        for i in range(10):
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
        for i in range(10):
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


cnn_model = CNN()

# Load the schedules
persona1, persona2 = cnn_model.load_schedules("output.xlsx")
print(persona1.num_weeks, "num_weeks")
print(persona1.display_schedule(1, 52))

stacked_schedules = Persona.stack_schedules(persona1, persona2, persona1.num_weeks)

stacked_schedules = np.array(stacked_schedules)

overlap_matrix = Persona.check_overlap(stacked_schedules)
stacked_schedules_transform = Persona.transform_matrix(stacked_schedules)
overlap_matrix_transform = Persona.transform_matrix(overlap_matrix)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(stacked_schedules_transform, overlap_matrix_transform,
                                                    test_size=0.2, random_state=42)
learning_rate = 0.001
cnn_model.train(X_train, y_train, batch_size=32, epochs=450, learning_rate=learning_rate)

# Make predictions using the trained model
predictions = cnn_model.model.predict(X_test)

week_to_predict = 11

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

week_to_predictValidation = 99  # Week three is at index 2
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

cnn_model.display_overlaps_info(week_to_predictValidation, persona1,
                                predicted_overlaps_classesValidation,actual_overlapsValidation)