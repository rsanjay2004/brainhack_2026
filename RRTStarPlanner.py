import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

class RRTStarPlanner:
    """
    RRT* path planner for continuous 2D space with point-cloud obstacles.
    Uses KDTree for fast collision checking and path smoothing for drone-ready waypoints.
    """
    def __init__(self, safety_margin=0.6, step_size=1.0, goal_radius=0.5,
                 rewire_radius=3.0, max_iter=3000, goal_bias=0.05):
        self.safety_margin = safety_margin
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.rewire_radius = rewire_radius
        self.max_iter = max_iter
        self.goal_bias = goal_bias

    def plan(self, start, goal, obstacle_points, bounds=None):
        start, goal = np.array(start), np.array(goal)
        if bounds is None:
            if len(obstacle_points) > 0:
                mins = obstacle_points.min(axis=0) - 2.0
                maxs = obstacle_points.max(axis=0) + 2.0
                bounds = np.array([mins, maxs]).T
            else:
                bounds = np.array([start-5, goal+5]).T

        tree_nodes = [start.copy()]
        costs = [0.0]
        parents = [-1]
        
        kdtree = KDTree(obstacle_points) if len(obstacle_points) > 0 else None

        for _ in range(self.max_iter):
            # 1. Sample
            if np.random.rand() < self.goal_bias:
                rand_node = goal.copy()
            else:
                rand_node = np.random.uniform(bounds[:, 0], bounds[:, 1])

            # 2. Nearest neighbor
            tree_arr = np.array(tree_nodes)
            nearest_idx = np.argmin(np.linalg.norm(tree_arr - rand_node, axis=1))
            nearest_node = tree_nodes[nearest_idx]

            # 3. Steer
            direction = rand_node - nearest_node
            dist = np.linalg.norm(direction)
            if dist == 0: continue
            new_node = nearest_node + (direction / dist) * self.step_size

            # 4. Collision check (single point)
            if kdtree is not None:
                min_dist, _ = kdtree.query(new_node, k=1)
                if min_dist < self.safety_margin:
                    continue

            # 5. Find neighbors & choose best parent
            dists_to_new = np.linalg.norm(tree_arr - new_node, axis=1)
            neighbor_indices = np.where(dists_to_new < self.rewire_radius)[0]
            
            min_cost = np.inf
            best_parent_idx = nearest_idx
            for i in neighbor_indices:
                if self._is_edge_free(tree_nodes[i], new_node, kdtree, self.safety_margin):
                    new_cost = costs[i] + np.linalg.norm(new_node - tree_nodes[i])
                    if new_cost < min_cost:
                        min_cost = new_cost
                        best_parent_idx = i

            # Add node
            tree_nodes.append(new_node)
            costs.append(min_cost)
            parents.append(best_parent_idx)
            new_idx = len(tree_nodes) - 1

            # 6. Rewire
            for i in neighbor_indices:
                if i == best_parent_idx: continue
                new_cost_via_new = min_cost + np.linalg.norm(new_node - tree_nodes[i])
                if new_cost_via_new < costs[i]:
                    if self._is_edge_free(new_node, tree_nodes[i], kdtree, self.safety_margin):
                        costs[i] = new_cost_via_new
                        parents[i] = new_idx

            # 7. Goal check
            if np.linalg.norm(new_node - goal) < self.goal_radius:
                tree_nodes.append(goal)
                costs.append(min_cost + np.linalg.norm(goal - new_node))
                parents.append(new_idx)
                raw_path = self._trace_path(tree_nodes, parents, len(tree_nodes)-1)
                return self._smooth_path(raw_path, kdtree, self.safety_margin)

        return None  # Failed

    def _is_edge_free(self, p1, p2, kdtree, margin):
        if kdtree is None: return True
        steps = max(3, int(np.linalg.norm(p2 - p1) / (margin * 0.4)))
        for t in np.linspace(0, 1, steps):
            pt = p1 + t * (p2 - p1)
            if kdtree.query(pt, k=1)[0] < margin:
                return False
        return True

    def _trace_path(self, nodes, parents, end_idx):
        path = []
        curr = end_idx
        while curr != -1:
            path.append(nodes[curr])
            curr = parents[curr]
        return np.array(path[::-1])

    def _smooth_path(self, path, kdtree, margin):
        """Shortcut smoothing: remove unnecessary waypoints"""
        if len(path) <= 2: return path
        smoothed = [path[0]]
        curr = 0
        while curr < len(path) - 1:
            next_idx = curr + 1
            while next_idx < len(path):
                if not self._is_edge_free(path[curr], path[next_idx], kdtree, margin):
                    break
                next_idx += 1
            smoothed.append(path[next_idx-1])
            curr = next_idx - 1
        return np.array(smoothed)