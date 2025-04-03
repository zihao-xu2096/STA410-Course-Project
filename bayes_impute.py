import numpy as np
from scipy.stats import norm
import pandas as pd
from numpy.random import normal
class BayesianImputer:
    
    def __init__(self, data, mechanism='MAR', priors=None, num_imputations=5, method='gibbs'):
        """
        Initializer of BayesianImputer.

        Parameters:
        - data: pandas.DataFrame
            The dataset containing missing values.
        - mechanism: str
            The assumed missing data mechanism. Must be 'MCAR', 'MAR', or 'MNAR'.(may remove MNAR)
        - priors: dict or None
            Optional dictionary of prior distributions for model parameters.
        - num_imputations: int
            Number of multiple imputations to generate.
        - method: str
            MCMC method to use: 'gibbs' or 'metropolis'.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        if mechanism not in ['MCAR', 'MAR', 'MNAR']:
            #May remove MNAR later
            raise ValueError("mechanism must be one of: 'MCAR', 'MAR', 'MNAR'.")

        if method not in ['gibbs', 'metropolis']:
            raise ValueError("method must be either 'gibbs' or 'metropolis'.")

        self.data = data.copy()
        self.mechanism = mechanism
        self.method = method
        self.num_imputations = num_imputations
        # self.priors = priors or {}  #Use empty dictionary if no priors provided
        self.missing_mask = self.data.isnull()
        self.imputed_datasets = []       #To store multiple imputed datasets
        self.trace_logs = []             #To store MCMC traces for diagnostics
        # self.posterior_samples = {}      #Posterior samples of model parameters
        self.initialized = False         #Whether missing values have been initialized
        self.model_fitted = False        #Whether the imputation has been completed
        self.initialized_data = []
        print(f"[Init] BayesianImputer initialized with method='{self.method}', "
              f"mechanism='{self.mechanism}', imputations={self.num_imputations}")

    def detect_missing_mechanism(self):
        """
        Placeholder for missingness mechanism detection.
        In practice, detecting mechanism from data is very difficult.
        Here we just summarize missingness patterns.
        """
        missing_summary = self.data.isnull().sum()
        print("[Detect] Missing value count per column:")
        print(missing_summary)
        return missing_summary

    def initialize_missing(self):
        """
        Simple initialization of missing values (mean imputation).
        Can be replaced by more sophisticated strategies later.
        """
        initialized_data = self.data.copy()
        for col in initialized_data.columns:
            if initialized_data[col].isnull().any():
                mean_val = initialized_data[col].mean()
                initialized_data[col] = initialized_data[col].fillna(mean_val)  # <- SAFE
        self.initialized = True
        self.initialized_data = initialized_data
        print("[InitMissing] Missing values initialized using column means.")

    def impute(self, return_all=True):
        """
        Perform Bayesian imputation using the selected MCMC method.
        Returns:
            - list of imputed datasets if return_all=True
            - single imputed dataset (first one) if return_all=False
        """
        if not self.initialized:
            raise RuntimeError("Missing values must be initialized before imputation.")

        print(f"[Impute] Running {self.method} sampling...")
        if self.method == 'gibbs':
            self._gibbs_sampler()
        elif self.method == 'metropolis':
            self._metropolis_hastings()

        self.model_fitted = True
        print("[Impute] Imputation complete.")

        return self.imputed_datasets if return_all else self.imputed_datasets[0]

    def _gibbs_sampler(self):
        """
        Simple Gibbs sampler for normally distributed data.
        Assumes missing-at-random and normal priors.
        """


        data = self.initialized_data.copy()
        missing_mask = self.missing_mask
        n_iter = 500
        imputed_versions = []

        print("[Gibbs] Starting Gibbs sampling...")
        for i in range(self.num_imputations):
            current_data = data.copy()
            trace = []

            for iter_ in range(n_iter):
                for col in current_data.columns:
                    if missing_mask[col].any():
                        observed = current_data[~missing_mask[col]][col]
                        mu = observed.mean()
                        sigma = observed.std()

                        # Sample new values for missing entries
                        sampled_values = normal(loc=mu, scale=sigma, size=missing_mask[col].sum())
                        current_data.loc[missing_mask[col], col] = sampled_values
                        trace.append((col, sampled_values))

            self.imputed_datasets.append(current_data.copy())
            self.trace_logs.append(trace)
            print(f"[Gibbs] Imputation {i+1} completed.")

        print("[Gibbs] Gibbs sampling finished.")

    def _metropolis_hastings(self):
        """
        Simple Metropolis-Hastings sampler for missing values.
        Assumes data is approximately normally distributed.
        Uses a random walk proposal for each missing value.
        """


        data = self.initialized_data.copy()
        missing_mask = self.missing_mask
        n_iter = 100
        imputed_versions = []

        print("[MH] Starting Metropolis-Hastings sampling...")

        for i in range(self.num_imputations):
            current_data = data.copy()
            trace = []

            for iter_ in range(n_iter):
                for col in current_data.columns:
                    if missing_mask[col].any():
                        observed = current_data[~missing_mask[col]][col]
                        mu = observed.mean()
                        sigma = observed.std()

                        # Propose new values using a random walk for each missing entry
                        current_missing = current_data.loc[missing_mask[col], col]
                        proposed = current_missing + np.random.normal(0, 0.1, size=current_missing.shape)

                        # Compute acceptance ratio (assuming Normal likelihood and symmetric proposal)
                        log_p_current = norm.logpdf(current_missing, loc=mu, scale=sigma).sum()
                        log_p_proposed = norm.logpdf(proposed, loc=mu, scale=sigma).sum()
                        acceptance_ratio = np.exp(log_p_proposed - log_p_current)

                        # Accept or reject
                        uniform_draws = np.random.rand(len(current_missing))
                        accepted = uniform_draws < acceptance_ratio

                        # Update accepted values
                        new_values = current_missing.copy()
                        new_values[accepted] = proposed[accepted]
                        current_data.loc[missing_mask[col], col] = new_values
                        trace.append((col, new_values))

            self.imputed_datasets.append(current_data.copy())
            self.trace_logs.append(trace)
            print(f"[MH] Imputation {i+1} completed.")

        print("[MH] Metropolis-Hastings sampling finished.")

    def generate_multiple_imputations(self):
        pass

    def posterior_diagnostics(self):
        pass

    def compare_with_pymc(self, pymc_results):
        pass

    def visualize_imputations(self):
        pass

    def simulate_missing(self, mechanism='MCAR', percent=0.1):
        pass

    def evaluate_overfitting(self):
        pass

    def set_priors(self, custom_priors):
        pass

    def causal_impute(self):
        pass
    
