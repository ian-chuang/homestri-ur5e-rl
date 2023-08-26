import math



def half_circle_with_equal_arc_lengths(radius, num_points):
    if num_points < 2:
        raise ValueError("The number of points must be at least 2.")

    # Calculate the angle increment to achieve equally spaced arc lengths
    angle_increment = math.pi / (num_points - 1)

    # Generate the Cartesian coordinates for each point
    cartesian_coordinates = []
    for i in range(num_points)[::-1]:
        angle = i * angle_increment
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        cartesian_coordinates.append((x + radius, y))

    return cartesian_coordinates

# Example usage:
radius = 0.25
num_points = 20
half_circle_points = half_circle_with_equal_arc_lengths(radius, num_points)

for point in half_circle_points:
    print(f"{point[0]} {point[1]} 0")