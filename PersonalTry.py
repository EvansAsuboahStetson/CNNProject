"""Create a persona class which Persona that handles scheduling for individuals.
 The set_schedule method sets the schedule for a given week, day, and availability,
  and the display_schedule method displays the schedule for a specified range of weeks. """
import math
from datetime import datetime, timedelta

import random

import numpy as np
import pandas as pd

import ast


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
            for _ in range(self.num_weeks + 1)
        ]
        # set the days of the persona
        self.days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        # set the color dictionary of the persona
        self.color_dict = {
            'free': (255, 255, 255),
            'not_available': (0, 0, 0),
            'somehow_available': (128, 128, 128)
        }

    def get_name(self):
        return self.name

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

            # set the start index of the persona for 30-minute intervals
        start_index = math.ceil((start_time - self.start_hour) * 60 / 30)
        # set the end index of the persona for 30-minute intervals
        end_index = math.ceil((end_time - self.start_hour) * 60 / 30)

        # set the persona schedule
        for i in range(start_index, end_index):
            if availability in self.color_dict:
                # set the persona schedule
                self.schedule[week][i][self.days.index(day)] = self.color_dict[availability]
            else:
                self.schedule[week][i][self.days.index(day)] = availability

    def display_schedule(self, start_week, end_week):
        # set the schedule of the persona
        for week in range(start_week, end_week + 1):
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
    def stack_schedules(num_weeks, *personas):
        stacked_schedules = []
        for week in range(num_weeks):
            week_schedules = []
            for persona in personas:
                week_schedules.append(persona.schedule[week])
            stacked_matrix = np.vstack(week_schedules)
            stacked_schedules.append(stacked_matrix)
        return stacked_schedules

    @staticmethod
    def generate_random_persona(name, start_hour, end_hour, interval_minutes, morning_preference,
                                day_preferences, days_to_avoid, num_weeks):
        # Instantiate a new Persona object
        persona = Persona(name, start_hour, end_hour, interval_minutes, num_weeks)

        for w in range(num_weeks):
            for day in persona.days:
                if day in days_to_avoid:
                    continue  # Skip the day if it is in the list of days to avoid
                for i in range(persona.num_slots):
                    # Calculate the hour based on the iteration
                    hour = start_hour + i * interval_minutes / 60

                    # Adjust probabilities and time ranges based on preferences
                    if 9 <= hour < 12 and morning_preference > 0:
                        rand_avail = random.choices(['free', 'somehow_available', 'not_available'],
                                                    weights=[morning_preference, 1, 1], k=1)[0]
                    elif day in day_preferences and 12 <= hour < 14 and day_preferences[day] > 0:
                        rand_avail = random.choices(['free', 'somehow_available', 'not_available'],
                                                    weights=[1, 1, day_preferences[day]], k=1)[0]
                    elif 12 <= hour < 14:
                        rand_avail = \
                            random.choices(['free', 'somehow_available', 'not_available'], weights=[3, 3, 2], k=1)[0]
                    elif 14 <= hour < 17:
                        rand_avail = \
                            random.choices(['free', 'somehow_available', 'not_available'], weights=[1, 2, 3], k=1)[0]
                    else:
                        rand_avail = 'not_available'

                    # Set the schedule for the persona
                    persona.set_schedule(w, day, hour, hour + interval_minutes / 60, rand_avail)

        # Return the generated persona
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

                        transformed_matrix[i, j, k] = 1
                    elif np.array_equal(matrix[i, j, k], [0, 0, 0]):

                        transformed_matrix[i, j, k] = 0
                    elif np.array_equal(matrix[i, j, k], [255, 255, 255]):

                        transformed_matrix[i, j, k] = 1
                    else:
                        transformed_matrix[i, j, k] = -1  # Default value for other cases
        return transformed_matrix

    @staticmethod
    def generate_random_schedule_for_year(persona_name, start_hour, end_hour, interval_minutes, morining_pref, day_pref,
                                          days_avoid, num_weeks=52):

        persona = Persona.generate_random_persona(persona_name, start_hour, end_hour, interval_minutes, morining_pref,
                                                  day_pref, days_avoid, num_weeks)
        return persona

    @staticmethod
    def save_schedules_to_excel(file_name, *personas):
        data = {'Week': [], 'Day': [], 'Time': []}
        for persona_num, persona in enumerate(personas, start=1):
            data[f'Persona{persona_num}'] = []
        for week in range(personas[0].num_weeks):
            for day in personas[0].days:
                for i in range(personas[0].num_slots):
                    time = personas[0].start_hour + i * personas[0].interval_minutes / 60
                    time_str = f"{int(time)}:{str(int((time % 1) * 60)).zfill(2)}"
                    data['Week'].append(week + 1)
                    data['Day'].append(day)
                    data['Time'].append(time_str)

                    for persona_num, persona in enumerate(personas, start=1):
                        data[f'Persona{persona_num}'].append(persona.schedule[week][i][persona.days.index(day)])

        df = pd.DataFrame(data)
        writer = pd.ExcelWriter(file_name, engine='openpyxl')
        df.to_excel(writer, index=False)
        writer.close()

    @staticmethod
    def load_schedules_from_excelTest(file_path, output_file='output2.xlsx', *persona_columns):
        df = pd.read_excel(file_path, nrows=4600)
        print(df.columns)

        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        print(persona_columns)
        persona_data = df[['Week', 'Day', 'Time', *persona_columns]]
        print(persona_data)

        for persona_num, columns in enumerate(persona_columns, start=1):
            print(columns, persona_num, "Hey")
            persona_data = df[['Week', 'Day', 'Time', columns]]
            sheet_name = f'Persona{persona_num}'
            persona_data.to_excel(writer, sheet_name=sheet_name, index=False)

        writer.close()

    @staticmethod
    def load_schedules_from_excel():
        file_path = "schedule2.xlsx"  # Replace with the actual path to your Excel file
        df = pd.read_excel(file_path, skiprows=range(1, 4601), nrows=10001 - 4601)

        # Replace 'Persona1', 'Persona2' with the actual persona column names in your Excel file
        persona_column_names = ['Persona1', 'Persona2', 'Persona3', 'Persona4']  # Add more persona names as needed

        # Create a dictionary to store DataFrames for each persona
        persona_data_dict = {}

        for persona_name in persona_column_names:
            persona_data = df[['Week', 'Day', 'Time', persona_name]]
            persona_data_dict[persona_name] = persona_data

        with pd.ExcelWriter('ValidationOutput2.xlsx', engine='xlsxwriter') as writer:
            # Write each DataFrame to a different worksheet
            for persona_name, persona_data in persona_data_dict.items():
                persona_data.to_excel(writer, sheet_name=persona_name, index=False)

    @staticmethod
    def set_schedules_from_load(file_path, *personas):
        xlsx = pd.ExcelFile(file_path)

        for persona in personas:
            sheet_name = persona.get_name()  # Adjust this according to how you store persona names
            df_persona = pd.read_excel(xlsx, sheet_name)

            for index, row in df_persona.iterrows():
                time_obj = datetime.strptime(row['Time'], '%H:%M')
                # Add an hour
                time_obj = time_obj + timedelta(hours=0.5)
                tuple_data = ast.literal_eval(row[sheet_name])  # Adjust this according to how you store persona data
                new_time_string = time_obj.strftime('%H:%M')

                persona.set_schedule(row['Week'], row['Day'], row['Time'], new_time_string, tuple_data)

        return personas

    @staticmethod
    def add_weeks(num_additional_weeks, *personas):
        for persona in personas:
            additional_schedule = [
                [[(0, 0, 0) for _ in range(len(persona.days))] for _ in range(persona.num_slots)]
                for _ in range(num_additional_weeks)
            ]
            persona.schedule.extend(additional_schedule)
            persona.num_weeks += num_additional_weeks

#
#
# # # add some days to avoid and day preferences. Make it vary according to personas
# #
# # # Create a persona object
# #
# persona1 = Persona.generate_random_schedule_for_year("Persona1", 8, 18, 30, 0.8,
#                                                      {'Monday': 0.5, 'Tuesday': 0.5, 'Wednesday': 0.5, 'Thursday': 0.5,
#                                                       'Friday': 0}, ['Friday'], 100)
# persona2 = Persona.generate_random_schedule_for_year("Persona2", 8, 18, 30, 0.2,
#                                                      {'Monday': 0.5, 'Tuesday': 0.5, 'Wednesday': 0.5, 'Thursday': 0,
#                                                       'Friday': 0.5}, ['Thursday'], 100)
# persona3 = Persona.generate_random_schedule_for_year("Persona3", 8, 18, 30, 0.5,
#                                                      {'Monday': 0.7, 'Tuesday': 0.7, 'Wednesday': 0.5, 'Thursday': 0.5,
#                                                       'Friday': 0}, ['Friday'], 100)
# persona4 = Persona.generate_random_schedule_for_year("Persona4", 8, 18, 30, 0.8,
#                                                      {'Monday': 0.4, 'Tuesday': 0.6, 'Wednesday': 0.5, 'Thursday': 0.3,
#                                                       'Friday': 0.2}, ['Friday'], 100)
# persona5 = Persona.generate_random_schedule_for_year("Persona5", 8, 18, 30, 0.2,
#                                                      {'Monday': 0.4, 'Tuesday': 0.6, 'Wednesday': 0.5, 'Thursday': 0.5,
#                                                       'Friday': 0.2}, ['Friday'], 100)
#
# Persona.save_schedules_to_excel("schedule2.xlsx", persona1, persona2, persona3, persona4, persona5)
#
# Persona.load_schedules_from_excelTest("schedule2.xlsx", "output2.xlsx", "Persona1", "Persona2", "Persona3", "Persona4",
#                                       "Persona5")
#
# Persona.load_schedules_from_excel()
#
#
#
#
#
#

