import numpy as np

# Create the matrices
matrix1 = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

matrix2 = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [1, 2, 3]])

# Stack the matrices
stacked_matrix = np.vstack((matrix1, matrix2))

# Track the overlap of values on the rows
overlap_rows = []
for i in range(len(matrix1)):
    if np.array_equal(np.sort(matrix1[i]), np.sort(matrix2[i])):
        overlap_rows.append(matrix1[i])

# Print the results
print("Matrix 1:")
print(matrix1)
print("\nMatrix 2:")
print(matrix2)
print("\nStacked Matrix:")
print(stacked_matrix)
print("\nOverlapping Rows:")
print(overlap_rows)
import numpy as np

# Create the matrices
matrix1 = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

matrix2 = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [1, 2, 3]])

# Stack the matrices
stacked_matrix = np.vstack((matrix1, matrix2))

# Track the overlap of values on the rows
overlap_rows = []
for i in range(len(matrix1)):
    if np.array_equal(np.sort(matrix1[i]), np.sort(matrix2[i])):
        overlap_rows.append(matrix1[i])

# Print the results
print("Matrix 1:")
print(matrix1)
print("\nMatrix 2:")
print(matrix2)
print("\nStacked Matrix:")
print(stacked_matrix)
print("\nOverlapping Rows:")
print(overlap_rows)
