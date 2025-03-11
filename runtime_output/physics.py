import numpy as np

class Physics:
    def __init__(self, normal_distance):
        self.normal_distance = normal_distance

    def adjust_normal_distance(self, positions, target_coverage=0.8, tolerance=0.05, adjustment_rate=0.01):
        lower_bounds = np.percentile(positions, 1, axis=0)
        upper_bounds = np.percentile(positions, 99, axis=0)
        width, height = upper_bounds - lower_bounds

        network_area = width * height

        # Adjust normal_distance based on coverage
        if network_area > target_coverage + tolerance:
            self.normal_distance *= (1 - adjustment_rate)  # Reduce normal_distance to shrink the network
        elif network_area < target_coverage - tolerance:
            self.normal_distance *= (1 + adjustment_rate)  # Increase normal_distance to expand the network

    def apply_forces(self, adjacency_matrix, positions, max_iterations=1):
        self.adjust_normal_distance(positions)
        speed = self.normal_distance

        for i in range(max_iterations):
            diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distances = np.sqrt(np.einsum('ijk,ijk->ij', diffs, diffs))
            normalized_directions = np.divide(diffs, distances[:, :, np.newaxis] + 1e-40)

            # Attraction/repulsion masks for connected nodes
            too_close = adjacency_matrix & (distances < 0.2 * self.normal_distance)
            too_far = adjacency_matrix & (distances > 0.3 * self.normal_distance)

            # Calculate forces for close and far connected nodes
            close_force = (0.2 * self.normal_distance - distances) * too_close
            far_force = (distances - 0.3 * self.normal_distance) * too_far

            # Repulsion for non-connected nodes
            within_range = ~adjacency_matrix & (distances < 1.7 * self.normal_distance)
            repulsion_force = within_range * np.divide((1.7 * self.normal_distance - distances), distances + 1e-40)

            # Apply forces
            forces = np.einsum('ijk,ij->ik', normalized_directions, close_force - far_force + repulsion_force)

            # Update positions based on forces
            adjustment = forces * speed
            positions += adjustment
            self.adjust_normal_distance(positions)
            positions = self.pull_all_nodes_towards_center(positions, speed)

            positions = np.clip(positions, 0, [500.0, 500.0])
            speed = self.normal_distance * 0.99 ** i

            if np.mean(np.abs(adjustment)) < 0.001:  # close enough
                break

        
        return positions

    def pull_all_nodes_towards_center(self, positions, force_strength):
        center = np.array([250, 250])
        diffs = center - positions
        positions += diffs * (force_strength / self.normal_distance)
        return positions
