import pyvista as pv
import numpy as np
import os

# Define the relative paths to the data files
data_dir = "data"
target_file = os.path.join(data_dir, "model.vtk")  # File containing points to check

# Load the target mesh
target_mesh = pv.read(target_file)

# Extract data arrays from the target mesh into NumPy arrays for faster indexing
target_x = np.array(target_mesh.points[:, 0])
target_y = np.array(target_mesh.points[:, 1])
target_z = np.array(target_mesh.points[:, 2])
target_data = np.column_stack((target_x, target_y, target_z))

# Initialize the list to store all filtered points with their Mask_values
all_filtered_points = []

# Process each cell file
counter = 0
for i in range(1, 2):
    name = os.path.join(data_dir, f"slice{i}.vtk")
    cell_mesh = pv.read(name)

    # Extract data arrays from the cell mesh into NumPy arrays
    cell_x = np.array(cell_mesh.points[:, 0])
    cell_y = np.array(cell_mesh.points[:, 1])
    cell_z = np.array(cell_mesh.points[:, 2])
    mask_values = np.array(cell_mesh['ScalarValue'])

    # Iterate over all cells in the current mesh
    for cell_index in range(cell_mesh.n_cells):
        # Get the current cell and its point IDs
        cell = cell_mesh.get_cell(cell_index)
        cell_point_ids = cell.point_ids

        # Collect data for the points in this cell
        cell_data = np.array([
            cell_x[cell_point_ids],
            cell_y[cell_point_ids],
            cell_z[cell_point_ids],
        ]).T  # Transpose to align each row with a point's data

        # Determine acceptable data ranges for the cell
        data_ranges = {
            "x": [cell_data[:, 0].min(), cell_data[:, 0].max()],
            "y": [cell_data[:, 1].min(), cell_data[:, 1].max()],
            "z": [cell_data[:, 2].min() - 5, cell_data[:, 2].max() + 5],
        }

        # Apply filtering in bulk using NumPy
        mask = (
            (target_data[:, 0] >= data_ranges["x"][0]) & (target_data[:, 0] <= data_ranges["x"][1]) &
            (target_data[:, 1] >= data_ranges["y"][0]) & (target_data[:, 1] <= data_ranges["y"][1]) &
            (target_data[:, 2] >= data_ranges["z"][0]) & (target_data[:, 2] <= data_ranges["z"][1])
        )

        # Append the filtered points and their corresponding Mask_value, cell_index, and file identifier
        filtered_points = target_data[mask]
        if filtered_points.size > 0:
            for point in filtered_points:
                all_filtered_points.append(list(point) + [mask_values[cell_index], counter])
			
        counter = counter + 1


# Convert to array for ML input
all_filtered_points_array = np.array(all_filtered_points)

# Specify the columns to check for duplicates
columns_indices = [0, 1, 2, 3]

# Extract the subset of columns to check for duplicates
subset_array = all_filtered_points_array[:, columns_indices]

# Find unique rows based on the specified columns
_, unique_indices = np.unique(subset_array, axis=0, return_index=True)

unique_indices = np.sort(unique_indices)

# Retain only the rows corresponding to unique values in the specified columns
unique_filtered_points = all_filtered_points_array[unique_indices, :]

print(f"Dataset after removing rows with duplicate columns {columns_indices} (total: {len(unique_filtered_points)}):")
print(unique_filtered_points)

output_file = os.path.join(data_dir, "Intersect_data")
np.save(output_file, unique_filtered_points)
print(f"Saved intersected data to {output_file}.npy")
