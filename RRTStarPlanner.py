import numpy as np


class _NNIndex:
    """Numpy brute-force nearest-neighbor index — drop-in for scipy KDTree (k=1 queries only)."""
    def __init__(self, points):
        self._pts = np.asarray(points, dtype=np.float32)

    def query(self, point, k=1):
        diffs = self._pts - np.asarray(point, dtype=np.float32)
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        idx = int(np.argmin(dists))
        return float(dists[idx]), idx


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
        start, goal = np.array(start, dtype=np.float32), np.array(goal, dtype=np.float32)
        if bounds is None:
            if len(obstacle_points) > 0:
                mins = obstacle_points.min(axis=0) - 2.0
                maxs = obstacle_points.max(axis=0) + 2.0
                bounds = np.array([mins, maxs]).T
            else:
                bounds = np.array([start-5, goal+5]).T

        # Pre-allocate fixed buffer — avoids O(n²) np.array(list) rebuild each iteration
        max_nodes = self.max_iter + 10
        tree_buf = np.empty((max_nodes, 2), dtype=np.float32)
        tree_buf[0] = start
        costs = np.full(max_nodes, np.inf, dtype=np.float64)
        costs[0] = 0.0
        parents = np.full(max_nodes, -1, dtype=np.int32)
        n_nodes = 1

        kdtree = _NNIndex(obstacle_points) if len(obstacle_points) > 0 else None

        for _ in range(self.max_iter):
            # 1. Sample
            if np.random.rand() < self.goal_bias:
                rand_node = goal.copy()
            else:
                rand_node = np.random.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float32)

            # 2. Nearest neighbor — use only populated slice
            tree_arr = tree_buf[:n_nodes]
            diffs = tree_arr - rand_node
            nearest_idx = int(np.argmin((diffs * diffs).sum(axis=1)))
            nearest_node = tree_buf[nearest_idx]

            # 3. Steer
            direction = rand_node - nearest_node
            dist = float(np.linalg.norm(direction))
            if dist == 0:
                continue
            new_node = (nearest_node + (direction / dist) * self.step_size).astype(np.float32)

            # 4. Collision check (single point)
            if kdtree is not None:
                min_dist, _ = kdtree.query(new_node, k=1)
                if min_dist < self.safety_margin:
                    continue

            # 5. Find neighbors & choose best parent
            dists_to_new = np.sqrt(((tree_arr - new_node) ** 2).sum(axis=1))
            neighbor_indices = np.where(dists_to_new < self.rewire_radius)[0]

            min_cost = np.inf
            best_parent_idx = nearest_idx
            for i in neighbor_indices:
                if self._is_edge_free(tree_buf[i], new_node, kdtree, self.safety_margin):
                    new_cost = costs[i] + float(np.linalg.norm(new_node - tree_buf[i]))
                    if new_cost < min_cost:
                        min_cost = new_cost
                        best_parent_idx = i

            if n_nodes >= max_nodes:
                break  # buffer full

            # Add node
            tree_buf[n_nodes] = new_node
            costs[n_nodes] = min_cost
            parents[n_nodes] = best_parent_idx
            new_idx = n_nodes
            n_nodes += 1

            # 6. Rewire
            for i in neighbor_indices:
                if i == best_parent_idx:
                    continue
                new_cost_via_new = min_cost + float(np.linalg.norm(new_node - tree_buf[i]))
                if new_cost_via_new < costs[i]:
                    if self._is_edge_free(new_node, tree_buf[i], kdtree, self.safety_margin):
                        costs[i] = new_cost_via_new
                        parents[i] = new_idx

            # 7. Goal check
            if float(np.linalg.norm(new_node - goal)) < self.goal_radius:
                # Add goal node
                goal_idx = n_nodes
                if goal_idx < max_nodes:
                    tree_buf[goal_idx] = goal
                    costs[goal_idx] = min_cost + float(np.linalg.norm(goal - new_node))
                    parents[goal_idx] = new_idx
                    n_nodes += 1
                raw_path = self._trace_path_buf(tree_buf, parents, goal_idx)
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

    def _trace_path_buf(self, tree_buf, parents, end_idx):
        path = []
        curr = int(end_idx)
        while curr != -1:
            path.append(tree_buf[curr].copy())
            curr = int(parents[curr])
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