import datetime
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Reshape, Conv2DTranspose
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PersonalTry import Persona


class CNN2:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        # Adjust the input shape based on your actual input data shape
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(80, 5, 3)))
        print(f"Shape after Conv2D: {model.output_shape}")

        model.add(MaxPooling2D((2, 1), strides=(2, 1), padding='same'))
        print(f"Shape after MaxPooling2D: {model.output_shape}")

        model.add(Conv2D(64, (3, 3), activation='sigmoid', padding='same'))
        print(f"Shape after Conv2D: {model.output_shape}")

        model.add(MaxPooling2D((2, 1), strides=(2, 1), padding='same'))
        print(f"Shape after MaxPooling2D: {model.output_shape}")

        model.add(Conv2D(128, (3, 3), activation='sigmoid', padding='same'))
        print(f"Shape after Conv2D: {model.output_shape}")

        model.add(MaxPooling2D((2, 1), strides=(2, 1), padding='same'))
        print(f"Shape after MaxPooling2D: {model.output_shape}")

        model.add(Conv2DTranspose(64, (3, 3), strides=(2, 1), padding='same'))
        print(f"Shape after Conv2DTranspose: {model.output_shape}")

        model.add(Conv2DTranspose(32, (3, 3), strides=(2, 1), padding='same'))
        print(f"Shape after Conv2DTranspose: {model.output_shape}")

        model.add(Conv2DTranspose(3, (3, 3), strides=(2, 1), activation='sigmoid', padding='same'))
        print(f"Final output shape: {model.output_shape}")

        model.summary()
        return model

    def train(self, input_data, output_data, batch_size, epochs, learning_rate, validation_data=None):
        print(input_data.shape)
        print(output_data.shape)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy',
                           metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                                    tf.keras.metrics.AUC()])

        # Check if the trained model file exists
        model_filename = "trained_model2.h5"
        if os.path.exists(model_filename):
            # Load the trained model
            self.model = load_model(model_filename)
        else:
            # Train the model with validation data
            history = self.model.fit(
                input_data, output_data,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data  # Pass your validation data here
            )

            print("Accuracy:", history.history['accuracy'])
            print("Precision:", history.history['precision'])
            print("Recall:", history.history['recall'])
            print("AUC:", history.history['auc'])
            print("Validation Accuracy:", history.history['val_accuracy'])  # New line for validation accuracy
            print("Validation Precision:", history.history['val_precision'])  # New line for validation precision
            print("Validation Recall:", history.history['val_recall'])  # New line for validation recall
            print("Validation AUC:", history.history['val_auc'])  # New line for validation AUC

            # Save the trained model
            self.model.save(model_filename)

            # Plot the loss curve
            self.plot_loss_curve(history)
    def plot_loss_curve(self, history):
        # Retrieve the loss and precision values from the history object
        loss = history.history['loss']
        precision = history.history['accuracy']

        epochs = range(1, len(loss) + 1)

        # Plot the loss curve
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'g', label='Training loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot the precision curve
        plt.subplot(1, 2, 2)
        plt.plot(epochs, precision, 'b', label='Training accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.legend()

        plt.show()

    @staticmethod
    def load_schedules(file_path, num_personas):
        persona_objects = [Persona(f"Persona{i}", 8, 18, 30, 52) for i in range(1, num_personas + 1)]

        return Persona.set_schedules_from_load(file_path, *persona_objects)

    @staticmethod
    def load_and_predict_validation(file_path, cnn_model):
        persona1, persona2 = cnn_model.load_schedules(file_path=file_path)
        stacked_schedules = Persona.stack_schedules(persona1, persona2, persona1.num_weeks)
        stacked_schedules = np.array(stacked_schedules)
        new_stacked_schedules_transform = Persona.transform_matrix(stacked_schedules)
        predictions_new = cnn_model.model.predict(new_stacked_schedules_transform)
        return predictions_new

    @staticmethod
    def plot_heatmap(predicted, actual, title="Overlap Visualization"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot Predicted Values
        im1 = axes[0].imshow(predicted[:, :, 0], cmap='coolwarm', interpolation='nearest', aspect='auto')
        axes[0].set_title('Predicted Overlaps')
        axes[0].set_xlabel('Days')
        axes[0].set_ylabel('Time Slots')
        fig.colorbar(im1, ax=axes[0], ticks=[0, 0.5, 1], label='Overlap Value')

        # Plot Actual Values
        im2 = axes[1].imshow(actual[:, :, 0], cmap='coolwarm', interpolation='nearest', aspect='auto')
        axes[1].set_title('Actual Overlaps')
        axes[1].set_xlabel('Days')
        axes[1].set_ylabel('Time Slots')
        fig.colorbar(im2, ax=axes[1], ticks=[0, 0.5, 1], label='Overlap Value')

        fig.suptitle(title)
        plt.show()

    @staticmethod
    def plot_overlaps_with_color(predicted_overlaps, actual_overlaps, week_number):
        def plot_subplot(ax, data, title):
            cmap = LinearSegmentedColormap.from_list('custom', [(0, 'black'), (0.5, 'white'), (1, 'blue')], N=256)
            im = ax.imshow(data[:, :, 0], cmap=cmap, interpolation='nearest', aspect='auto', vmin=0, vmax=1)
            ax.set_title(title)
            ax.set_xlabel('Days')
            ax.set_ylabel('Time Slots')
            for i in range(80):
                for j in range(5):
                    text = ax.text(j, i, f"{data[i, j, 0]:.2f}", ha="center", va="center", color="w", fontsize=8)

            return im

        # Reshape the predicted overlaps
        reshaped_predictions = predicted_overlaps.reshape((predicted_overlaps.shape[0], 80, 5, 3))

        # Reshape the actual overlaps
        reshaped_actual_overlaps = actual_overlaps.reshape(80, 5, 3)

        # Increase the figure size and adjust aspect ratio
        fig, axs = plt.subplots(1, 2, figsize=(45, 25),
                                gridspec_kw={'width_ratios': [3, 3]})  # Adjust the values as needed

        # Plot predicted overlaps
        im1 = plot_subplot(axs[0], reshaped_predictions[0], 'Predicted Overlaps')
        # Plot actual overlaps
        im2 = plot_subplot(axs[1], reshaped_actual_overlaps, 'Actual Overlaps')

        # Add the week number as a title
        fig.suptitle(f'Week {week_number}', fontsize=16)
        fig.subplots_adjust(wspace=0.4)  # Adjust width space between subplots

        # Add colorbars
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        fig.colorbar(im1, cax=cbar_ax, label='Value')

        plt.show()

    @staticmethod
    def map_to_discrete(predictions, thresholds=(0.3, 0.6)):
        """
        Map continuous predictions to discrete values based on specified thresholds.

        Parameters:
        - predictions: NumPy array of continuous predictions.
        - thresholds: Tuple of thresholds defining the ranges for discrete mapping.

        Returns:
        - Discrete predictions.
        """
        low_threshold, mid_threshold = thresholds

        discrete_predictions = np.zeros_like(predictions)
        discrete_predictions[(predictions >= low_threshold) & (predictions < mid_threshold)] = 1
        discrete_predictions[predictions >= mid_threshold] = 1
        return discrete_predictions

    @staticmethod
    def display_overlaps_info(week_to_predict, persona1, predicted_overlaps_classes, actual_overlapsValidation):
        print("\nDate and Time for Predicted OverlapsValidation:")
        for i in range(16):
            for j in range(5):
                if predicted_overlaps_classes[0, i, j, 0] != 0:
                    print(
                        f"Overlap at Week {week_to_predict}, Day {persona1.days[j]}, Time {persona1.start_hour + i * persona1.interval_minutes / 60:.2f}")
        print("\nDate and Time for Actual Overlaps:")
        for i in range(16):
            for j in range(5):
                if actual_overlapsValidation[i, j, 0] != 0:
                    print(
                        f"Overlap at Week {week_to_predict}, Day {persona1.days[j]}, Time {persona1.start_hour + i * persona1.interval_minutes / 60:.2f}")


cnn_model = CNN2()
np.set_printoptions(threshold=np.inf)

# Load the schedules
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


persona1 = Persona("Persona1", 8, 18, 30, 52)
week_to_predict = 23
X_to_predict = stacked_schedules_transform[week_to_predict].reshape(1, 80, 5, 3)
predictions = cnn_model.model.predict(X_to_predict)
overlap_matrix_transform = overlap_matrix_transform[week_to_predict]
discrete_predictions = cnn_model.map_to_discrete(predictions)

print(overlap_matrix_transform.shape)

print(discrete_predictions.shape)

flat_actual = overlap_matrix_transform.flatten()
flat_predictions = discrete_predictions.flatten()


## Filter values greater than 0.6
mask = flat_actual > 0.6
filtered_flat_actual = flat_actual[mask]
filtered_flat_predictions = flat_predictions[mask]


accuracy = accuracy_score(filtered_flat_actual, filtered_flat_predictions)
precision = precision_score(filtered_flat_actual, filtered_flat_predictions, average='weighted', zero_division=1)
recall = recall_score(filtered_flat_actual, filtered_flat_predictions, average='weighted', zero_division=1)
f1 = f1_score(filtered_flat_actual, filtered_flat_predictions, average='weighted', zero_division=1)

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Use the function to plot the heatmap
cnn_model.plot_overlaps_with_color(predictions, overlap_matrix_transform, week_to_predict)




Persona.add_weeks(48, *personas)
Persona.set_schedules_from_load("ValidationOutput2.xlsx", *personas)

stacked_schedulesValidation = Persona.stack_schedules(personas[0].num_weeks, *personas)
stacked_schedulesValidation = np.array(stacked_schedulesValidation)
overlap_matrixValidation = Persona.check_overlap(stacked_schedulesValidation)
stacked_schedules_transformValidation = Persona.transform_matrix(stacked_schedulesValidation)
overlap_matrix_transformValidation = Persona.transform_matrix(overlap_matrixValidation)



persona1 = Persona("Persona1", 8, 18, 30, 52)
week_to_predict = 67
X_to_predictValidation = stacked_schedules_transformValidation[week_to_predict].reshape(1, 80, 5, 3)
predictionsValidation = cnn_model.model.predict(X_to_predictValidation)
overlap_matrix_transformValidation = overlap_matrix_transformValidation[week_to_predict]
discrete_predictionsValidation = cnn_model.map_to_discrete(predictionsValidation)



print(overlap_matrix_transformValidation.shape)

print(discrete_predictionsValidation.shape)

cnn_model.plot_overlaps_with_color(predictionsValidation, overlap_matrix_transformValidation, week_to_predict)


flat_actual = overlap_matrix_transformValidation.flatten()
flat_predictions = cnn_model.map_to_discrete(predictionsValidation).flatten()

## Filter values greater than 0.6
mask = flat_actual > 0.6
filtered_flat_actual = flat_actual[mask]
filtered_flat_predictions = flat_predictions[mask]


accuracy = accuracy_score(filtered_flat_actual, filtered_flat_predictions)
precision = precision_score(filtered_flat_actual, filtered_flat_predictions, average='weighted', zero_division=1)
recall = recall_score(filtered_flat_actual, filtered_flat_predictions, average='weighted', zero_division=1)
f1 = f1_score(filtered_flat_actual, filtered_flat_predictions, average='weighted', zero_division=1)

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")