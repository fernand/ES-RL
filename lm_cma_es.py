"""
Limited Memory CMA-ES implementation from scratch using NumPy.
Based on "Limited-Memory Matrix Adaptation for Large Scale Black-Box Optimization"
by Loshchilov, Glasmachers, and Beyer (2017)

Fixed implementation addressing mathematical correctness issues.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple


class LMCMAES:
    """Limited Memory Covariance Matrix Adaptation Evolution Strategy"""

    def __init__(
        self,
        x0: np.ndarray,
        sigma: float = 0.5,
        inopts: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LM-CMA-ES optimizer.

        Args:
            x0: Initial solution vector
            sigma: Initial step size
            inopts: Options dictionary containing:
                - popsize: Population size (default: 4 + 3*log(n))
                - bounds: Optional bounds [lower, upper]
                - seed: Random seed
                - memory_limit: Memory budget m (default: 4 + 3*log(n))
        """
        # Set random seed if provided
        if inopts and 'seed' in inopts:
            np.random.seed(inopts['seed'])

        # Dimension and initial point
        self.n = len(x0)
        self.xmean = np.array(x0, dtype=np.float64)
        self.sigma = sigma

        # Parse options
        opts = inopts or {}

        # Population size
        self.lambda_ = opts.get('popsize', int(4 + 3 * np.log(self.n)))
        self.mu = self.lambda_ // 2  # Number of parents

        # Recombination weights
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mueff = 1.0 / np.sum(self.weights**2)

        # Limited memory settings
        self.m = opts.get('memory_limit', int(4 + 3 * np.log(self.n)))
        self.m = min(self.m, self.n)  # Can't exceed dimension

        # Step size control parameters
        self.c_sigma = opts.get('c_sigma', (self.mueff + 2) / (self.n + self.mueff + 5))
        self.d_sigma = opts.get('d_sigma', 1 + self.c_sigma + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1))

        # Evolution path for step size control
        self.p_sigma = np.zeros(self.n)

        # Limited memory Cholesky factors: A = prod_j (I + a_j * v_j * v_j^T)
        self.v = []  # Direction vectors (unit norm)
        self.a = []  # Scalars with a_i > -1

        # Learning rate for rank-one updates
        self.eta = min(0.5, 2.0 * self.mueff / (self.n + 2.0)) / self.n

        # Bounds
        self.bounds = opts.get('bounds', None)
        if self.bounds is not None:
            self.lower_bound = self.bounds[0]
            self.upper_bound = self.bounds[1]

        # Generation counter
        self.generation = 0

        # Expected value of ||N(0,I)||
        self.chiN = np.sqrt(self.n) * (1 - 1/(4*self.n) + 1/(21*self.n**2))

        # Storage for current population
        self.current_pop = None
        self.current_z = None  # Store z vectors for tell()

    def ask(self) -> np.ndarray:
        """
        Generate new population of candidate solutions.

        Returns:
            Array of shape (popsize, n) containing candidate solutions
        """
        self.current_pop = np.zeros((self.lambda_, self.n))
        self.current_z = np.zeros((self.lambda_, self.n))

        for i in range(self.lambda_):
            # Generate standard normal vector
            z = np.random.randn(self.n)
            self.current_z[i] = z

            # Apply Cholesky factor transformation: y = A * z
            y = self._transform_vector(z)

            # Create offspring
            x = self.xmean + self.sigma * y

            # Handle bounds with resampling for unbiased distribution
            if self.bounds is not None:
                max_resample = 100
                for _ in range(max_resample):
                    if np.all(x >= self.lower_bound) and np.all(x <= self.upper_bound):
                        break
                    # Resample if out of bounds
                    z = np.random.randn(self.n)
                    self.current_z[i] = z
                    y = self._transform_vector(z)
                    x = self.xmean + self.sigma * y
                # Final clip as fallback
                x = np.clip(x, self.lower_bound, self.upper_bound)

            self.current_pop[i] = x

        return self.current_pop

    def tell(self, solutions: np.ndarray, fitness_values: np.ndarray):
        """
        Update the optimizer with evaluated solutions.

        Args:
            solutions: Array of evaluated solutions
            fitness_values: Corresponding fitness values (to be minimized)
        """
        # Sort by fitness
        idx = np.argsort(fitness_values)

        # Select mu best solutions
        selected = solutions[idx[:self.mu]]
        selected_z = self.current_z[idx[:self.mu]]

        # Compute weighted mean of selected solutions
        xold = self.xmean.copy()
        self.xmean = np.dot(self.weights, selected)

        # Update step size using CSA
        self._update_step_size(selected, xold)

        # Update limited memory Cholesky factors
        self._update_memory(selected, xold)

        self.generation += 1

    def _transform_vector(self, z: np.ndarray) -> np.ndarray:
        """
        Apply Cholesky factor: y = A * z where A = prod_j (I + a_j * v_j * v_j^T)

        Args:
            z: Standard normal vector

        Returns:
            Transformed vector y
        """
        y = z.copy()
        # Sequential product, not sum!
        for v_i, a_i in zip(self.v, self.a):
            y += a_i * v_i * (v_i @ y)
        return y

    def _inverse_transform_vector(self, y: np.ndarray) -> np.ndarray:
        """
        Apply inverse of Cholesky factor: z = A^{-1} * y
        Using Sherman-Morrison formula: (I + a*v*v^T)^{-1} = I - (a/(1+a)) * v * v^T

        Args:
            y: Transformed vector

        Returns:
            Original vector z
        """
        z = y.copy()
        # Apply inverses in reverse order
        for v_i, a_i in reversed(list(zip(self.v, self.a))):
            z -= (a_i / (1.0 + a_i)) * v_i * (v_i @ z)
        return z

    def _update_step_size(self, selected: np.ndarray, xold: np.ndarray):
        """
        Update step size using Cumulative Step-size Adaptation (CSA).

        Args:
            selected: Selected (best) solutions
            xold: Previous mean
        """
        # Compute weighted mean step in original space
        y_w = (self.xmean - xold) / self.sigma

        # Transform to whitened space
        z_w = self._inverse_transform_vector(y_w)

        # Update evolution path
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
                      np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mueff) * z_w

        # Update step size
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) *
                            (np.linalg.norm(self.p_sigma) / self.chiN - 1))

    def _update_memory(self, selected: np.ndarray, xold: np.ndarray):
        """
        Update limited memory Cholesky factors using rank-μ update approximation.

        Args:
            selected: Selected (best) solutions
            xold: Previous mean
        """
        # Compute steps in original space
        y = (selected - xold) / self.sigma  # shape (mu, n)

        # Transform to whitened space
        z = np.vstack([self._inverse_transform_vector(y_i) for y_i in y])

        # Compute weighted recombination in whitened space
        z_w = np.dot(self.weights, z)  # shape (n,)

        # Compute norm
        nz = np.linalg.norm(z_w)
        if nz < 1e-12:
            return

        # Normalize to get direction vector
        v_new = z_w / nz

        # Compute scalar factor (centering at 1 in whitened space)
        a_new = self.eta * (nz**2 / self.n - 1.0)

        # Keep factor invertible
        a_new = max(a_new, -0.9)

        # Add to memory with FIFO policy
        if len(self.v) >= self.m:
            self.v.pop(0)
            self.a.pop(0)

        self.v.append(v_new)
        self.a.append(a_new)

    def stop(self) -> bool:
        """
        Check termination criteria.

        Returns:
            True if optimization should stop
        """
        # Simple termination criteria
        if self.generation > 1000:
            return True
        if self.sigma < 1e-10:
            return True
        return False

    @property
    def result(self) -> Tuple[np.ndarray, float]:
        """
        Get current best solution and step size.

        Returns:
            Tuple of (best solution, current step size)
        """
        return self.xmean, self.sigma

    def test_consistency(self) -> bool:
        """
        Test mathematical consistency of transform/inverse operations.

        Returns:
            True if transforms are consistent
        """
        z = np.random.randn(self.n)
        y = self._transform_vector(z)
        z_recovered = self._inverse_transform_vector(y)
        return np.allclose(z, z_recovered, atol=1e-8)
