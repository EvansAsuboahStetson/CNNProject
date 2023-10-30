"""Create a persona class which Persona that handles scheduling for individuals.
 The set_schedule method sets the schedule for a given week, day, and availability,
  and the display_schedule method displays the schedule for a specified range of weeks. """
from datetime import datetime, timedelta

import random

import numpy as np
import pandas as pd

import ast

from xlsxwriter import Workbook

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D


# create a class called persona
class Persona:
    # initialize the class
    def __init__(self, name, start_hour, end_hour, interval_minutes, num_weeks):
        # set the name of the persona
        self.name = name
        # set the start hour of the persona
        self.start_hour = start_hour
        # set the end hour of the persona
        self.end_hour = end_hour
        # set the interval minutes of the persona
        self.interval_minutes = interval_minutes
        # set the number of slots of the persona
        self.num_slots = int((end_hour - start_hour) * 60 / interval_minutes)
        # set the number of weeks of the persona
        self.num_weeks = num_weeks
        # set the schedule of the persona
        self.schedule = [
            [[(0, 0, 0) for _ in range(5)] for _ in range(self.num_slots)]
            for _ in range(self.num_weeks+1)
        ]
        # set the days of the persona
        self.days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        # set the color dictionary of the persona
        self.color_dict = {
            'free': (255, 255, 255),
            'not_available': (0, 0, 0),
            'somehow_available': (128, 128, 128)
        }

    @staticmethod
    def time_to_float(time_string):
        return float(time_string.replace(":", "."))


    # set the schedule of the persona
    def set_schedule(self, week, day, start_time, end_time, availability):

        if isinstance(start_time, str):

            start_time = self.time_to_float(start_time)
            end_time = self.time_to_float(end_time)

        if int(start_time) >= self.end_hour:
            return
        # set the start index of the persona
        start_index = int((start_time - self.start_hour) * 60 / self.interval_minutes)
        # set the end index of the persona
        end_index = int((end_time - self.start_hour) * 60 / self.interval_minutes)

        # set the persona schedule
        for i in range(start_index, end_index):
            if availability in self.color_dict:
                # set the persona schedule
                self.schedule[week][i][self.days.index(day)] = self.color_dict[availability]
            else:

                self.schedule[week][i][self.days.index(day)] =  availability

    def display_schedule(self, start_week, end_week):
        # set the schedule of the persona
        for week in range(start_week, end_week+1):
            # set the schedule of the persona
            print(f"Week {week} schedule:")
            # set the schedule of the persona
            for i in range(len(self.schedule[week])):
                # set the time of the persona
                time = self.start_hour + i * self.interval_minutes / 60
                # set the time string of the persona
                time_str = f"{int(time)}:{str(int((time % 1) * 60)).zfill(2)}"
                # set the schedule of the persona
                print(f"{time_str} - {self.schedule[week][i]}")
                # stack the schedules of the persona

    @staticmethod
    def stack_schedules(persona1, persona2, num_weeks):
        stacked_schedules = []
        for week in range(num_weeks):
            stacked_matrix = np.vstack((persona1.schedule[week], persona2.schedule[week]))
            stacked_schedules.append(stacked_matrix)
        return stacked_schedules

    @staticmethod
    def generate_random_persona(name, start_hour, end_hour, interval_minutes, num_weeks):
        # set the persona
        persona = Persona(name, start_hour, end_hour, interval_minutes, num_weeks)
        # set the persona
        for w in range(num_weeks):
            # set the persona
            for day in persona.days:
                # set the persona
                for i in range(persona.num_slots):
                    # set the hour of the persona
                    hour = start_hour + i * interval_minutes / 60
                    # set the probabilities and time ranges of the persona with weights for the random
                    if 9 <= hour < 12:
                        # set the random availability of the persona
                        rand_avail = \
                            random.choices(['free', 'somehow_available', 'not_available'], weights=[4, 3, 3], k=1)[0]
                    elif 12 <= hour < 14:

                        rand_avail = \
                            random.choices(['free', 'somehow_available', 'not_available'], weights=[3, 6, 1], k=1)[0]
                    elif 14 <= hour < 17:

                        rand_avail = \
                            random.choices(['free', 'somehow_available', 'not_available'], weights=[2, 3, 5], k=1)[0]
                    else:

                        rand_avail = 'not_available'

                    persona.set_schedule(w, day, hour, hour + interval_minutes / 60, rand_avail)
                    # return the persona
        return persona



    @staticmethod
    def print_stack_schedules(data):
        data = np.array(data)
        print("Stacked Schedules:")

        for i in range(data.shape[0]):

            for j in range(data.shape[1]):
                print(data[i, j], end=" ")

            print("\n")
        print("\n")

    @staticmethod
    def check_overlap(stacked_schedules):
        stacked_schedules = np.array(stacked_schedules)
        overlap_matrix = np.zeros_like(stacked_schedules)  # Matrix to store overlapping values
        for col in range(stacked_schedules.shape[1] - 8):  # Limit the range to handle the 7 index shift
            col1 = stacked_schedules[:, col, :]
            col2 = stacked_schedules[:, col + 8, :]  # Change the index of col2
            for i in range(col1.shape[0]):
                for j in range(col1.shape[1]):
                    if np.array_equal(col1[i, j], col2[i, j]):
                        # print(f"Cell at ({i}, {j}) in col1 and col2 are the same: {col1[i, j]} and {col2[i, j]}")
                        overlap_matrix[i, col, j] = col1[i, j]  # Storing the overlapping value in the matrix
        return overlap_matrix

    @staticmethod
    def transform_matrix(matrix):
        matrix = np.array(matrix)
        transformed_matrix = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                for k in range(matrix.shape[2]):
                    if np.array_equal(matrix[i, j, k], [128, 128, 128]):
                        transformed_matrix[i, j, k] = 2
                    elif np.array_equal(matrix[i, j, k], [0, 0, 0]):
                        transformed_matrix[i, j, k] = 0
                    elif np.array_equal(matrix[i, j, k], [255, 255, 255]):
                        transformed_matrix[i, j, k] = 1
                    else:
                        transformed_matrix[i, j, k] = -1  # Default value for other cases
        return transformed_matrix

    @staticmethod
    def generate_random_schedule_for_year(persona_name, start_hour, end_hour, interval_minutes, num_weeks=52):

        persona = Persona.generate_random_persona(persona_name, start_hour, end_hour, interval_minutes, num_weeks)
        return persona

    @staticmethod
    def save_schedules_to_excel(persona1, persona2, file_name):
        data = {'Week': [], 'Day': [], 'Time': [], 'Persona1': [], 'Persona2': []}
        for week in range(persona1.num_weeks):
            for day in persona1.days:
                for i in range(persona1.num_slots):
                    time = persona1.start_hour + i * persona1.interval_minutes / 60
                    time_str = f"{int(time)}:{str(int((time % 1) * 60)).zfill(2)}"
                    data['Week'].append(week + 1)
                    data['Day'].append(day)
                    data['Time'].append(time_str)
                    data['Persona1'].append(persona1.schedule[week][i][persona1.days.index(day)])
                    data['Persona2'].append(persona2.schedule[week][i][persona1.days.index(day)])

        df = pd.DataFrame(data)
        writer = pd.ExcelWriter(file_name, engine='openpyxl')
        df.to_excel(writer, index=False)
        writer.close()

    @staticmethod
    def load_schedules_from_excel():
        file_path = "schedules.xlsx"  # Replace with the actual path to your Excel file
        df = pd.read_excel(file_path, skiprows=range(1, 2081), nrows=4001 - 2081)

        persona1_data = df[['Week', 'Day', 'Time', 'Persona1']]
        persona2_data = df[['Week', 'Day', 'Time', 'Persona2']]

        with pd.ExcelWriter('ValidationOutput.xlsx', engine='xlsxwriter') as writer:
            # Write each DataFrame to a different worksheet
            persona1_data.to_excel(writer, sheet_name='Persona1', index=False)
            persona2_data.to_excel(writer, sheet_name='Persona2', index=False)


    @staticmethod
    def setSchedulesFromLoad(file_path, persona1, persona2):

        file_path = file_path
        xlsx = pd.ExcelFile(file_path)
        df_persona1 = pd.read_excel(xlsx, 'Persona1')
        df_persona2 = pd.read_excel(xlsx, 'Persona2')

        for index, row in df_persona1.iterrows():
            time_obj = datetime.strptime(row['Time'], '%H:%M')
            # Add an hour
            time_obj = time_obj + timedelta(hours=1)
            tuple_data = ast.literal_eval(row['Persona1'])

            # Convert the datetime object back to string format
            new_time_string = time_obj.strftime('%H:%M')

            persona1.set_schedule(row['Week'], row['Day'], row['Time'], new_time_string, tuple_data)

        # Set schedule for Persona2
        for index, row in df_persona2.iterrows():
            time_obj = datetime.strptime(row['Time'], '%H:%M')
            # Add an hour
            time_obj = time_obj + timedelta(hours=1)
            tuple_data = ast.literal_eval(row['Persona2'])

            # Convert the datetime object back to string format
            new_time_string = time_obj.strftime('%H:%M')

            persona2.set_schedule(row['Week'], row['Day'], row['Time'], new_time_string, tuple_data)
        return persona1, persona2

    def add_weeks(self, num_additional_weeks):
        additional_schedule = [
            [[(0, 0, 0) for _ in range(5)] for _ in range(self.num_slots)]
            for _ in range(num_additional_weeks)
        ]
        self.schedule.extend(additional_schedule)
        self.num_weeks += num_additional_weeks







