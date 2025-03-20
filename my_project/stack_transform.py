import torch

def generate_pancake_coordinates(x, y, z, radius=0.045, height=0.010, flag='stack', stack_matrix=None):
    if stack_matrix is None:
        raise ValueError("stack_matrix must be provided.")
    
    # Get input dimensions
    rows, cols = stack_matrix.shape
    
    # Initialize an empty list to store coordinates
    coordinates = []
    
    # Iterate through each column
    for col in range(cols):
        z_offset = height * col
        y_offset = radius * col

        # Iterate through each row
        for row in range(rows):
            # Compute the x coordinate with the correct offset
            x_offset = (row - (rows - 1) / 2) * 2 * radius
            x_current = x + x_offset

            # Compute the current y and z coordinates
            if flag == 'stack':
                y_current = y
                z_current = z + z_offset
            elif flag == 'flat':
                y_current = y + y_offset
                z_current = z
            else:
                raise ValueError("flag must be 'stack' or 'flat'.")

            # Add to the coordinates list
            coordinates.append((x_current, y_current, z_current))
    
    # Convert the coordinates to a torch tensor
    result_matrix = torch.tensor(coordinates)
    
    return result_matrix

# Example usage
x, y, z = 0.0, 0.0, 0.0
radius = 0.045
height = 0.010

# Case 1: Input shape (2, 3)
stack_matrix_2x3 = torch.zeros((2, 3))
result_2x3 = generate_pancake_coordinates(x, y, z, radius, height, flag='stack', stack_matrix=stack_matrix_2x3)
print("Result for 2x3 input (stack):")
print(result_2x3)

# Case 2: Input shape (3, 3)
stack_matrix_3x3 = torch.zeros(3, 3)
result_3x3 = generate_pancake_coordinates(x, y, z, radius, height, flag='flat', stack_matrix=stack_matrix_3x3)
print("Result for 3x3 input (flat):")
print(result_3x3)

# Uncomment the following cases to test other scenarios
# Case 3: Input shape (4, 2)
stack_matrix_4x2 = torch.zeros(4, 2)
result_4x2 = generate_pancake_coordinates(x, y, z, radius, height, flag='stack', stack_matrix=stack_matrix_4x2)
print("Result for 4x2 input (stack):")
print(result_4x2)

# Case 4: Input shape (5, 4)
stack_matrix_5x4 = torch.zeros(5, 4)
result_5x4 = generate_pancake_coordinates(x, y, z, radius, height, flag='flat', stack_matrix=stack_matrix_5x4)
print("Result for 5x4 input (flat):")
print(result_5x4)