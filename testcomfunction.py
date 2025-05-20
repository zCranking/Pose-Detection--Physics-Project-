import numpy as np

def calculate_com(joint_positions, total_mass=70):
    """
    Calculate the center of mass (CoM) based on joint positions and estimated masses.
    
    Parameters:
    joint_positions (dict): A dictionary where each key is a joint name and each value is a (x, y) tuple.
    total_mass (float): Total mass of the person (default 70 kg).
    
    Returns:
    tuple: (com_x, com_y, total_mass, mass_distribution)
    """
    # Define bone connections (roughly following the human body structure)
    bone_pairs = {
        "head": ("head", "torso"),
        "left_arm": ("left_shoulder", "left_elbow"),
        "right_arm": ("right_shoulder", "right_elbow"),
        "left_forearm": ("left_elbow", "left_wrist"),
        "right_forearm": ("right_elbow", "right_wrist"),
        "left_leg": ("left_hip", "left_knee"),
        "right_leg": ("right_hip", "right_knee"),
        "left_shin": ("left_knee", "left_ankle"),
        "right_shin": ("right_knee", "right_ankle"),
        "torso": ("torso", "pelvis")
    }

    # Calculate segment lengths
    segment_masses = {}
    total_length = 0.0
    
    for segment, (joint1, joint2) in bone_pairs.items():
        if joint1 in joint_positions and joint2 in joint_positions:
            # Calculate Euclidean distance
            p1 = np.array(joint_positions[joint1])
            p2 = np.array(joint_positions[joint2])
            length = np.linalg.norm(p1 - p2)
            segment_masses[segment] = length
            total_length += length
    
    # Scale lengths to estimate masses
    for segment in segment_masses:
        segment_masses[segment] = (segment_masses[segment] / total_length) * total_mass
    
    # Calculate the center of mass
    com_x = 0.0
    com_y = 0.0
    for segment, (joint1, joint2) in bone_pairs.items():
        if joint1 in joint_positions and joint2 in joint_positions:
            # Use the midpoint of the segment as the mass center
            p1 = np.array(joint_positions[joint1])
            p2 = np.array(joint_positions[joint2])
            midpoint = (p1 + p2) / 2
            mass = segment_masses[segment]
            com_x += midpoint[0] * mass
            com_y += midpoint[1] * mass
    
    # Final CoM position
    com_x /= total_mass
    com_y /= total_mass
    
    # Calculate mass distribution
    mass_distribution = {segment: mass / total_mass for segment, mass in segment_masses.items()}
    
    return com_x, com_y, total_mass, mass_distribution

# Example usage
joint_positions = {
    "head": (0.5, 0.9),
    "torso": (0.5, 0.6),
    "pelvis": (0.5, 0.4),
    "left_shoulder": (0.4, 0.7),
    "right_shoulder": (0.6, 0.7),
    "left_elbow": (0.3, 0.6),
    "right_elbow": (0.7, 0.6),
    "left_wrist": (0.2, 0.5),
    "right_wrist": (0.8, 0.5),
    "left_hip": (0.4, 0.4),
    "right_hip": (0.6, 0.4),
    "left_knee": (0.4, 0.2),
    "right_knee": (0.6, 0.2),
    "left_ankle": (0.4, 0.0),
    "right_ankle": (0.6, 0.0),
}

com_x, com_y, total_mass, mass_distribution = calculate_com(joint_positions)

print(f"CoM: ({com_x:.2f}, {com_y:.2f})")
print(f"Total Mass: {total_mass:.2f} kg")
print("Mass Distribution:")
for segment, fraction in mass_distribution.items():
    print(f"  {segment}: {fraction:.2f} kg")