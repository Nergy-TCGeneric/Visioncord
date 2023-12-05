# Known values of RPi camera v2.
HORIZONTAL_FOV = 62.2
VERTICAL_FOV = 48.8
ACTIVE_H_PIXELS_MAX = 3280
ACTIVE_V_PIXELS_MAX = 2464

def calculate_angle(image_size: "tuple[int, int]", point: "tuple[int, int]") -> "tuple[float, float]":
    # Simple calculation based on center point is a point with zero degree angle
    center_x, center_y = image_size[0] // 2, image_size[1] // 2
    angle_x = (point[0] - center_x) / image_size[0] * HORIZONTAL_FOV
    angle_y = (point[1] - center_y) / image_size[1] * VERTICAL_FOV
    return (angle_x, angle_y)