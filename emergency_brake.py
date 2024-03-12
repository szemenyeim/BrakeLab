class EmergencyBrake:
    """
        This class defines an EmergencyBrake object that evaluates the distance of objects
        and triggers a braking action if deemed necessary. It takes into account minimum
        distance thresholds in the x, y, and z directions, along with a security margin.
    """
    def __init__(self, **kwargs):
        # Initialize the EmergencyBrake object with default parameters.
        # If provided, override defaults with values from kwargs.
        self.min_x = kwargs.pop("min_x", 0.15)  # Minimum distance in the x-direction
        self.min_y = kwargs.pop("min_y", 0.15)  # Minimum distance in the y-direction
        self.min_z = kwargs.pop("min_z", 0.4)  # Minimum distance in the z-direction
        self.security_margin = kwargs.pop("security_margin", 0.15)  # Security margin

    def __call__(self, object_list: list) -> bool:
        # Check distance for each object in the object_list and store the results.
        results = [object.check_distance(self.min_x, self.min_y, self.min_z, self.security_margin) for object in
                   object_list]

        # Count the number of False results, indicating objects within unsafe distance.
        false_count = results.count(False)

        # If there are fewer than 2 objects within unsafe distance, return True (safe), otherwise False (unsafe).
        return false_count < 2
