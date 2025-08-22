"""
Limited Memory CMA-ES implementation from scratch using NumPy.
Based on the paper: "Limited-Memory Matrix Adaptation for Large Scale Black-Box Optimization"
by Loshchilov, Glasmachers, and Beyer (2017)
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
                - c_sigma: Cumulation constant for step size control
                - d_sigma: Damping for step size control
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
        
        # Adaptation rates
        self.c_sigma = opts.get('c_sigma', (self.mueff + 2) / (self.n + self.mueff + 5))
        self.d_sigma = opts.get('d_sigma', 1 + self.c_sigma + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1))
        
        # Evolution path for step size control
        self.p_sigma = np.zeros(self.n)
        
        # Limited memory vectors and scalars
        self.v = []  # Direction vectors
        self.d = []  # Scaling factors
        
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
        
    def ask(self) -> np.ndarray:
        """
        Generate new population of candidate solutions.
        
        Returns:
            Array of shape (popsize, n) containing candidate solutions
        """
        self.current_pop = np.zeros((self.lambda_, self.n))
        
        for i in range(self.lambda_):
            # Generate random vector
            z = np.random.randn(self.n)
            
            # Apply limited memory transformation
            y = self._transform_vector(z)
            
            # Create offspring
            x = self.xmean + self.sigma * y
            
            # Apply bounds if specified
            if self.bounds is not None:
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
        
        # Compute weighted mean of selected solutions
        xold = self.xmean.copy()
        self.xmean = np.dot(self.weights, selected)
        
        # Compute evolution path for step size adaptation
        y_w = (self.xmean - xold) / self.sigma
        
        # Transform back to get z_w
        z_w = self._inverse_transform_vector(y_w)
        
        # Update evolution path
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
                      np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mueff) * z_w
        
        # Update step size
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * 
                            (np.linalg.norm(self.p_sigma) / self.chiN - 1))
        
        # Update limited memory vectors
        self._update_memory(selected, xold)
        
        self.generation += 1
    
    def _transform_vector(self, z: np.ndarray) -> np.ndarray:
        """
        Transform standard normal vector using limited memory representation.
        
        Args:
            z: Standard normal vector
            
        Returns:
            Transformed vector y = M^(1/2) * z
        """
        y = z.copy()
        
        # Apply stored transformations
        for i, (v_i, d_i) in enumerate(zip(self.v, self.d)):
            # y = y + sqrt(d_i) * (v_i^T * z) * v_i
            y += np.sqrt(max(0, d_i)) * np.dot(v_i, z) * v_i
        
        return y
    
    def _inverse_transform_vector(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform using limited memory representation.
        
        Args:
            y: Transformed vector
            
        Returns:
            Original vector z such that y = M^(1/2) * z
        """
        z = y.copy()
        
        # Apply inverse transformations in reverse order
        for v_i, d_i in reversed(list(zip(self.v, self.d))):
            if d_i > 0:
                # z = z - sqrt(d_i)/(1 + d_i) * (v_i^T * z) * v_i
                factor = np.sqrt(d_i) / (1 + d_i)
                z -= factor * np.dot(v_i, z) * v_i
        
        return z
    
    def _update_memory(self, selected: np.ndarray, xold: np.ndarray):
        """
        Update limited memory vectors and scalars.
        
        Args:
            selected: Selected (best) solutions
            xold: Previous mean
        """
        # Compute weighted covariance update direction
        c_mu = min(1, 2 * self.mueff / self.n)
        
        for i in range(self.mu):
            # Compute difference vector
            diff = (selected[i] - xold) / self.sigma
            
            # Get normalized direction
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                v_new = diff / norm
                d_new = self.weights[i] * c_mu * (norm**2 - 1)
                
                # Check if we should add this to memory
                if abs(d_new) > 1e-10:
                    # Add to memory (with limit)
                    if len(self.v) >= self.m:
                        # Remove oldest
                        self.v.pop(0)
                        self.d.pop(0)
                    
                    self.v.append(v_new)
                    self.d.append(d_new)
        
        # Decay old directions
        decay_factor = 0.99
        self.d = [d * decay_factor for d in self.d]
        
        # Remove directions with very small eigenvalues
        indices_to_keep = [i for i, d in enumerate(self.d) if abs(d) > 1e-10]
        self.v = [self.v[i] for i in indices_to_keep]
        self.d = [self.d[i] for i in indices_to_keep]
    
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