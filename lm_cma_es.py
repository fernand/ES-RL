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

        # Bounds (support scalar bounds by broadcasting)
        self.bounds = opts.get('bounds', None)
        if self.bounds is not None:
            self.lower_bound = np.asarray(self.bounds[0]) + np.zeros(self.n)
            self.upper_bound = np.asarray(self.bounds[1]) + np.zeros(self.n)

        # Generation counter
        self.generation = 0

        # Expected value of ||N(0,I)||
        self.chiN = np.sqrt(self.n) * (1 - 1/(4*self.n) + 1/(21*self.n**2))

        # Storage for current population
        self.current_pop = None
        self.current_z = None  # Store z vectors for tell()
        
        # Track best solution so far
        self.best_x = self.xmean.copy()
        self.best_f = np.inf

    def ask(self) -> np.ndarray:
        """
        Generate new population of candidate solutions using mirrored sampling.

        Returns:
            Array of shape (popsize, n) containing candidate solutions
        """
        self.current_pop = np.zeros((self.lambda_, self.n))
        self.current_z = np.zeros((self.lambda_, self.n))
        self.valid_mask = np.ones(self.lambda_, dtype=bool)  # Track valid samples

        # Mirrored sampling for variance reduction
        for i in range(0, self.lambda_, 2):
            # Generate base random vector
            z = np.random.randn(self.n)
            
            # Process both z and -z (mirrored)
            for sign, idx in [(+1, i), (-1, min(i+1, self.lambda_-1))]:
                if idx >= self.lambda_:
                    break
                    
                z_signed = sign * z
                self.current_z[idx] = z_signed
                
                # Apply Cholesky factor transformation: y = A * z
                y = self._transform_vector(z_signed)
                
                # Create offspring
                x = self.xmean + self.sigma * y
                
                # Handle bounds with resampling for unbiased distribution
                if self.bounds is not None:
                    max_resample = 100
                    valid = False
                    
                    for attempt in range(max_resample):
                        if np.all(x >= self.lower_bound) and np.all(x <= self.upper_bound):
                            valid = True
                            break
                        # Resample if out of bounds
                        z_new = np.random.randn(self.n)
                        self.current_z[idx] = z_new
                        y = self._transform_vector(z_new)
                        x = self.xmean + self.sigma * y
                    
                    if not valid:
                        # Use reflection as fallback instead of clipping
                        x = np.where(x < self.lower_bound, 
                                    2 * self.lower_bound - x, x)
                        x = np.where(x > self.upper_bound, 
                                    2 * self.upper_bound - x, x)
                        # Mark as invalid for update exclusion
                        self.valid_mask[idx] = False
                
                self.current_pop[idx] = x

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

        # Track best solution
        if fitness_values[idx[0]] < self.best_f:
            self.best_f = fitness_values[idx[0]]
            self.best_x = solutions[idx[0]].copy()
        
        # Select mu best solutions (excluding invalid ones if needed)
        selected = solutions[idx[:self.mu]]
        valid_selected = self.valid_mask[idx[:self.mu]]
        
        # Compute weighted mean of selected solutions
        xold = self.xmean.copy()
        # Down-weight or exclude invalid samples
        adjusted_weights = self.weights.copy()
        adjusted_weights[~valid_selected] *= 0.1  # Heavily down-weight invalid
        adjusted_weights /= adjusted_weights.sum()
        self.xmean = np.dot(adjusted_weights, selected)

        # Update step size using CSA
        self._update_step_size(selected, xold)

        # Update limited memory Cholesky factors (only with valid samples)
        if np.any(valid_selected):
            valid_indices = np.where(valid_selected)[0]
            selected_valid = selected[valid_indices]
            weights_valid = self.weights[valid_indices] / np.sum(self.weights[valid_indices])
            self._update_memory_with_weights(selected_valid, xold, weights_valid)
        
        # Occasional re-normalization to fight FP drift
        if self.generation % 50 == 0 and len(self.v) > 0:
            for i in range(len(self.v)):
                norm = np.linalg.norm(self.v[i])
                if abs(norm - 1.0) > 1e-6:
                    self.v[i] /= norm

        self.generation += 1
        
        # Periodic consistency check for debugging
        if self.generation % 50 == 0 and self.generation > 0:
            if not self.test_consistency():
                import warnings
                warnings.warn(f"Transform consistency degraded at generation {self.generation}")

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

    def _update_memory_with_weights(self, selected: np.ndarray, xold: np.ndarray, weights: np.ndarray):
        """
        Update limited memory Cholesky factors with custom weights.
        
        Args:
            selected: Selected (best) solutions
            xold: Previous mean
            weights: Custom weights for selected solutions
        """
        # Compute steps in original space
        y = (selected - xold) / self.sigma  # shape (mu, n)
        
        # Transform to whitened space
        z = np.vstack([self._inverse_transform_vector(y_i) for y_i in y])
        
        # Compute weighted recombination in whitened space
        z_w = np.dot(weights, z)  # shape (n,)
        
        # Compute norm
        nz = np.linalg.norm(z_w)
        if nz < 1e-12:
            return
        
        # Normalize to get direction vector
        v_new = z_w / nz
        
        # Compute scalar factor (centering at 1 in whitened space)
        a_new = self.eta * (nz**2 / self.n - 1.0)
        
        # Clip to keep factor stable and invertible
        a_new = np.clip(a_new, -0.9, 2.0)  # Keep 1+a in [0.1, 3.0]
        
        # Add to memory with FIFO policy
        if len(self.v) >= self.m:
            self.v.pop(0)
            self.a.pop(0)
        
        self.v.append(v_new)
        self.a.append(a_new)
    
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

        # Clip to keep factor stable and invertible
        a_new = np.clip(a_new, -0.9, 2.0)  # Keep 1+a in [0.1, 3.0]

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
        Get best solution found so far and current step size.

        Returns:
            Tuple of (best solution, current step size)
        """
        return self.best_x, self.sigma

    def test_consistency(self, verbose: bool = False) -> bool:
        """
        Test mathematical consistency of transform/inverse operations.
        
        Args:
            verbose: If True, print detailed error information

        Returns:
            True if transforms are consistent
        """
        # Test multiple random vectors
        n_tests = min(10, self.n)
        max_error = 0.0
        
        for _ in range(n_tests):
            z = np.random.randn(self.n)
            y = self._transform_vector(z)
            z_recovered = self._inverse_transform_vector(y)
            error = np.linalg.norm(z - z_recovered) / (np.linalg.norm(z) + 1e-10)
            max_error = max(max_error, error)
        
        if verbose:
            print(f"Transform consistency check: max relative error = {max_error:.2e}")
            if max_error > 1e-6:
                print("WARNING: Transform consistency degraded!")
        
        return max_error < 1e-6
