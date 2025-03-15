import math
import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (in 2D space).
    
    Args:
        a: First point [x, y]
        b: Mid point [x, y] - this is the vertex of the angle
        c: End point [x, y]
    
    Returns:
        Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Handle numerical errors that might cause cosine_angle to be slightly outside [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Calculate angle in degrees
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

def calculate_distance(a, b):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        a: First point [x, y]
        b: Second point [x, y]
    
    Returns:
        Distance between points
    """
    a = np.array(a)
    b = np.array(b)
    
    return np.linalg.norm(a - b)

def is_point_above(point, reference):
    """
    Determine if a point is above another point in image coordinates (y-axis is inverted).
    
    Args:
        point: Point to check [x, y]
        reference: Reference point [x, y]
    
    Returns:
        Boolean indicating if point is above reference
    """
    return point[1] < reference[1]

def calculate_body_alignment(shoulders, hips, knees):
    """
    Calculate how well-aligned parts of the body are.
    
    Args:
        shoulders: Midpoint between shoulders [x, y]
        hips: Midpoint between hips [x, y]
        knees: Midpoint between knees [x, y]
    
    Returns:
        Deviation angle from vertical alignment
    """
    # For a perfect vertical alignment, these points should be on a straight line
    return calculate_angle(shoulders, hips, knees)

def calculate_midpoint(a, b):
    """
    Calculate the midpoint between two points.
    
    Args:
        a: First point [x, y]
        b: Second point [x, y]
    
    Returns:
        Midpoint [x, y]
    """
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]

def normalize_screen_coordinates(landmarks, image_width, image_height):
    """
    Normalize landmarks to actual pixel coordinates.
    
    Args:
        landmarks: List of landmarks with normalized coordinates
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        List of landmarks with pixel coordinates
    """
    normalized = []
    for landmark in landmarks:
        normalized.append([
            int(landmark[0] * image_width),
            int(landmark[1] * image_height)
        ])
    return normalized
