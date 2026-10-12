"""
Applicability Domain Module for Materials Informatics
Project: Machine-Learning Prediction of Mechanical Properties of Steels from Composition
IIT Roorkee - B.Tech Project in Metallurgical and Materials Engineering
Date: 

Defines a physically principled, statistically audited Applicability Domain (AD) estimator:
1. Elemental bounding box (min / max per constituent in wt%)
2. Compositional distance to nearest training sample:
   - Excludes Iron (Fe) because Fe = 100 - sum(solutes) adds no independent degree of freedom.
   - Evaluates both standardized non-Fe distances and raw non-Fe Euclidean distances.
3. Domain membership classification with strict, documented rules.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class ApplicabilityDomain:
    """
    Quantifies the domain of applicability for composition-based alloy property models.
    
    A query alloy is considered 'Inside Domain' if and only if:
    1. Bounding Box Rule: All elemental concentrations lie within [min_i - tol, max_i + tol]
       of the training set distribution.
    2. Proximity Rule: The distance to the nearest training sample (d_min) in standardized
       non-solvent (non-Fe) composition space does not exceed the cutoff threshold
       (by default, the 95th percentile of training nearest-neighbor distances).
    """

    def __init__(self, metric='euclidean', distance_percentile=95.0, 
                 box_tolerance=0.0, exclude_element='Fe', standardize=True):
        """
        Parameters
        ----------
        metric : str
            Distance metric for nearest neighbor search ('euclidean').
        distance_percentile : float
            Percentile of training nearest-neighbor distance used as cutoff (default 95.0).
        box_tolerance : float
            Allowable fractional tolerance beyond the min/max box (default 0.0, strict).
        exclude_element : str or None
            Solvent matrix element to exclude from distance calculation (default 'Fe').
        standardize : bool
            Whether to standardize non-solvent features before computing distances (default True).
        """
        self.metric = metric
        self.distance_percentile = distance_percentile
        self.box_tolerance = box_tolerance
        self.exclude_element = exclude_element
        self.standardize = standardize
        
        self.feature_names = None
        self.active_indices_ = None
        self.active_feature_names_ = None
        self.min_bounds_ = None
        self.max_bounds_ = None
        
        self.scaler_ = None
        self.nn_model_ = None
        self.X_train_raw_ = None
        self.X_train_active_ = None
        self.train_nn_distances_ = None
        self.distance_cutoff_ = None

    def fit(self, X, feature_names=None):
        """
        Fit the applicability domain to the training alloy compositions.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Training feature matrix in weight percent (wt%).
        feature_names : list of str, optional
            Names of the alloying elements.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_arr = X.values.astype(float)
        else:
            X_arr = np.asarray(X, dtype=float)
            self.feature_names = feature_names if feature_names else [f"El_{i}" for i in range(X_arr.shape[1])]

        self.X_train_raw_ = X_arr
        
        # 1. Bounding box computation across all features (including Fe)
        self.min_bounds_ = np.min(X_arr, axis=0)
        self.max_bounds_ = np.max(X_arr, axis=0)
        
        # 2. Identify active features for distance calculation (excluding solvent matrix Fe)
        if self.exclude_element and self.exclude_element in self.feature_names:
            self.active_indices_ = [i for i, name in enumerate(self.feature_names) if name != self.exclude_element]
        else:
            self.active_indices_ = list(range(len(self.feature_names)))
            
        self.active_feature_names_ = [self.feature_names[i] for i in self.active_indices_]
        X_active = X_arr[:, self.active_indices_]
        
        # 3. Standardization (if enabled)
        if self.standardize:
            self.scaler_ = StandardScaler()
            self.X_train_active_ = self.scaler_.fit_transform(X_active)
        else:
            self.scaler_ = None
            self.X_train_active_ = X_active
            
        # 4. Fit Nearest Neighbors on active space (k=2 for self-distance audit)
        self.nn_model_ = NearestNeighbors(n_neighbors=2, metric=self.metric)
        self.nn_model_.fit(self.X_train_active_)
        
        dists, _ = self.nn_model_.kneighbors(self.X_train_active_)
        self.train_nn_distances_ = dists[:, 1]
        
        # 5. Distance cutoff
        self.distance_cutoff_ = np.percentile(self.train_nn_distances_, self.distance_percentile)
        
        return self

    def query(self, X_query):
        """
        Evaluate domain membership for candidate/test alloy compositions.

        Parameters
        ----------
        X_query : np.ndarray or pd.DataFrame
            Query alloy feature matrix in wt%.

        Returns
        -------
        results : pd.DataFrame
            Detailed domain diagnostics per sample:
            - inside_box: bool (all elements within training bounds)
            - nearest_dist_std: float (distance in standardized non-Fe space)
            - nearest_dist_raw_wt: float (distance in raw non-Fe wt% space)
            - inside_distance: bool (nearest_dist_std <= distance_cutoff)
            - inside_domain: bool (inside_box and inside_distance)
            - num_box_violations: int
            - violating_elements: list of str
        """
        if isinstance(X_query, pd.DataFrame):
            X_arr = X_query.values.astype(float)
        else:
            X_arr = np.asarray(X_query, dtype=float)
            
        # 1. Check Box Bounds (full composition including Fe)
        tol_min = self.min_bounds_ - self.box_tolerance * np.abs(self.min_bounds_)
        tol_max = self.max_bounds_ + self.box_tolerance * np.abs(self.max_bounds_)
        
        below_min = X_arr < tol_min
        above_max = X_arr > tol_max
        violations = below_min | above_max
        
        inside_box = ~np.any(violations, axis=1)
        num_violations = np.sum(violations, axis=1)
        
        violating_elements = []
        for row in violations:
            viols = [self.feature_names[i] for i, v in enumerate(row) if v]
            violating_elements.append(viols)

        # 2. Extract active features (non-Fe)
        X_q_active = X_arr[:, self.active_indices_]
        
        # Standardized distance query
        if self.scaler_ is not None:
            X_q_std = self.scaler_.transform(X_q_active)
        else:
            X_q_std = X_q_active
            
        nn_std = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(self.X_train_active_)
        std_dists, std_indices = nn_std.kneighbors(X_q_std)
        std_dists = std_dists.flatten()
        std_indices = std_indices.flatten()
        
        # Raw wt% distance query for comparison
        raw_active_train = self.X_train_raw_[:, self.active_indices_]
        nn_raw = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(raw_active_train)
        raw_dists, _ = nn_raw.kneighbors(X_q_active)
        raw_dists = raw_dists.flatten()
        
        inside_distance = std_dists <= self.distance_cutoff_
        inside_domain = inside_box & inside_distance
        
        results = pd.DataFrame({
            'inside_box': inside_box,
            'nearest_dist_std': std_dists,
            'nearest_dist_raw_wt': raw_dists,
            'nearest_train_idx': std_indices,
            'inside_distance': inside_distance,
            'inside_domain': inside_domain,
            'num_box_violations': num_violations,
            'violating_elements': violating_elements
        })
        
        return results

    def get_summary_dict(self):
        """Return dictionary of domain boundary statistics."""
        return {
            'n_features_total': len(self.feature_names),
            'n_features_active': len(self.active_feature_names_),
            'excluded_element': self.exclude_element,
            'standardized': self.standardize,
            'n_train_samples': len(self.X_train_raw_),
            'median_train_nn_dist_std': float(np.median(self.train_nn_distances_)),
            'p95_train_nn_dist_std': float(np.percentile(self.train_nn_distances_, 95.0)),
            'p99_train_nn_dist_std': float(np.percentile(self.train_nn_distances_, 99.0)),
            'distance_cutoff': float(self.distance_cutoff_),
            'distance_percentile': self.distance_percentile,
            'bounds': {
                el: {'min_wt': float(self.min_bounds_[i]), 'max_wt': float(self.max_bounds_[i])}
                for i, el in enumerate(self.feature_names)
            }
        }
