import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
        # self.missing_mask = self.data.isnull()
        # self.imputed_datasets = []       #To store multiple imputed datasets
        # self.trace_logs = []             #To store MCMC traces for diagnostics
        # self.posterior_samples = {}      #Posterior samples of model parameters
        # self.initialized = False         #Whether missing values have been initialized
        # self.model_fitted = False        #Whether the imputation has been completed

        print(f"[Init] BayesianImputer initialized with method='{self.method}', "
              f"mechanism='{self.mechanism}', imputations={self.num_imputations}")

    def detect_missing_mechanism(self):
        pass

    def initialize_missing(self):
        pass

    def impute(self, return_all=True):
        pass

    def _gibbs_sampler(self):
        pass

    def _metropolis_hastings(self):
        pass

    def generate_multiple_imputations(self):
        pass

    def posterior_diagnostics(self):
        """
        Plot diagnostic graphs for MCMC convergence and posterior stability.

        Assumes `self.trace_logs` is a dictionary where keys are parameter names
        and values are lists or arrays of sampled values from the MCMC process.

        For each parameter, the following are plotted:
        - Trace plot (sample values vs iteration)
        - Histogram (posterior distribution)

        Raises:
        - ValueError: if `self.trace_logs` is empty or uninitialized.
        """
        if not self.trace_logs:
            raise ValueError("No MCMC trace logs found. Run imputation to collect posterior samples.")

        for param, samples in self.trace_logs.items():
            plt.figure(figsize=(14, 5))

            # Trace Plot
            plt.subplot(1, 2, 1)
            plt.plot(samples, color="blue", linewidth=0.7)
            plt.title(f"Trace Plot: {param}")
            plt.xlabel("Iteration")
            plt.ylabel("Sample Value")

            # Histogram of posterior
            plt.subplot(1, 2, 2)
            sns.histplot(samples, kde=True, bins=30, color="orange")
            plt.title(f"Posterior Distribution: {param}")
            plt.xlabel("Value")
            plt.ylabel("Density")

            plt.suptitle(f"Posterior Diagnostics for '{param}'")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

    def compare_with_pymc(self, pymc_results):
        pass

    def visualize_imputations(self, kind="kde", show_summary=True, max_cols=4):
        """
        Visualize distributions of observed and imputed values for each variable with missing data.

        This function provides visual diagnostics to assess the quality of imputed values. For each
        column with missing data, it compares the observed values to the imputed values aggregated
        across all multiple imputations (if available), mimicking PyMC's posterior distribution plots.

        Parameters:
        - kind: str
            Type of plot: 'kde' (density estimation) or 'hist' (histogram).
        - show_summary: bool
            If True, overlays vertical lines for observed and imputed means.
        - max_cols: int
            Maximum number of subplots per row.

        Raises:
        - ValueError: if imputed_datasets is empty or invalid.
        """

        if not self.imputed_datasets or not isinstance(self.imputed_datasets, list):
            raise ValueError("No imputed datasets available. Run impute() or generate_multiple_imputations() first.")

        print("[Visualize Imputations] Generating plots for observed vs. imputed distributions...")

        n_imputations = len(self.imputed_datasets)
        imputed_concat = pd.concat(self.imputed_datasets, keys=range(n_imputations), names=['imputation'])
        missing_mask = self.data.isnull()

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        n_plots = sum(self.data[col].isnull().any() for col in self.data.columns)

        if n_plots == 0:
            print("[Visualize Imputations] No missing values found in any column.")
            return

        n_rows = int(np.ceil(n_plots / max_cols))
        fig, axes = plt.subplots(n_rows, min(max_cols, n_plots), figsize=(5 * max_cols, 4 * n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
        plot_idx = 0

        for col in self.data.columns:
            if not self.data[col].isnull().any():
                continue

            ax = axes[plot_idx]
            observed = self.data[col].dropna()
            imputed_all = imputed_concat.loc[(slice(None),), col]
            imputed_only = imputed_concat.loc[(slice(None), self.data[col].isnull()), col]

            if col in numeric_cols:
                # Numeric columns: use KDE or histogram
                if kind == "kde":
                    sns.kdeplot(observed, ax=ax, color='blue', label='Observed', linewidth=2)
                    sns.kdeplot(imputed_only, ax=ax, color='orange', label='Imputed', linewidth=2)
                else:
                    sns.histplot(observed, ax=ax, color='blue', label='Observed', bins=30, stat='density')
                    sns.histplot(imputed_only, ax=ax, color='orange', label='Imputed', bins=30, stat='density')

                if show_summary:
                    ax.axvline(observed.mean(), color='blue', linestyle='--', alpha=0.7, label='Obs Mean')
                    ax.axvline(imputed_only.mean(), color='orange', linestyle='--', alpha=0.7, label='Imp Mean')

                ax.set_ylabel("Density")
            else:
                # Categorical columns: use bar plots
                observed_counts = observed.value_counts(normalize=True)
                imputed_counts = imputed_only.value_counts(normalize=True)
                categories = sorted(set(observed_counts.index).union(imputed_counts.index))

                obs_vals = [observed_counts.get(cat, 0) for cat in categories]
                imp_vals = [imputed_counts.get(cat, 0) for cat in categories]

                x = np.arange(len(categories))
                ax.bar(x - 0.2, obs_vals, width=0.4, label='Observed', color='blue')
                ax.bar(x + 0.2, imp_vals, width=0.4, label='Imputed', color='orange')
                ax.set_xticks(x)
                ax.set_xticklabels(categories)

                ax.set_ylabel("Proportion")

            ax.set_title(f"Observed vs Imputed: {col}")
            ax.legend()
            plot_idx += 1

        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])  # Remove unused subplots

        plt.tight_layout()
        plt.show()
    def simulate_missing(self, mechanism='MCAR', percent=0.1):
        """
        Simulate missingness in the dataset according to a specified mechanism.

        Parameters:
        - mechanism: str
            One of 'MCAR', 'MAR'. MNAR is not yet implemented.
        - percent: float
            Proportion of total entries to set as missing (0 < percent < 1).

        This method modifies `self.data` and stores a copy of the original complete
        data in `self.original_data` for evaluation purposes.
        """
        if not 0 < percent < 1:
            raise ValueError("percent must be a float between 0 and 1.")

        self.original_data = self.data.copy()

        n_rows, n_cols = self.data.shape
        total_cells = n_rows * n_cols
        n_missing = int(total_cells * percent)

        data_copy = self.data.copy()

        if mechanism == 'MCAR':
            # Randomly choose cells to make missing
            missing_indices = list(zip(
                np.random.randint(0, n_rows, n_missing),
                np.random.randint(0, n_cols, n_missing)
            ))
            for row, col in missing_indices:
                data_copy.iat[row, col] = np.nan

        elif mechanism == 'MAR':
            # MAR: missingness in one column depends on values in another
            if n_cols < 2:
                raise ValueError("MAR simulation requires at least 2 columns.")

            # Choose two different columns
            col_a, col_b = np.random.choice(data_copy.columns, 2, replace=False)
            threshold = data_copy[col_a].median()

            # Rows where col_a is above median will have missing in col_b
            mar_rows = data_copy[data_copy[col_a] > threshold].sample(
                frac=percent, random_state=42
            )
            data_copy.loc[mar_rows.index, col_b] = np.nan

        else:
            raise NotImplementedError("Only 'MCAR' and 'MAR' are currently supported.")

        self.data = data_copy
        self.missing_mask = self.data.isnull()

        print(f"[Simulate Missing] Applied {mechanism} mechanism with {percent*100:.1f}% missingness.")

    def evaluate_overfitting(self):
        """
        Evaluate potential overfitting in the imputation process by comparing imputed values
        to the original (true) values from the dataset with simulated missingness.

        Assumes that the original full dataset is stored in `self.original_data`
        and that `self.imputed_datasets` contains multiple imputed versions of the data.

        Calculates:
        - RMSE between imputed and true values
        - Average width of posterior intervals (if multiple imputations are available)
        - Coverage of true values within 95% credible intervals (if applicable)

        Raises:
        - ValueError: if original data or imputations are missing.
        """
        if not hasattr(self, 'original_data'):
            raise ValueError("Original data not found. Use simulate_missing() and store original data first.")
        if not self.imputed_datasets:
            raise ValueError("No imputed datasets found. Run generate_multiple_imputations() first.")

        results = {}

        # Convert imputations into a 3D array: (num_imputations, num_rows, num_columns)
        imputed_array = np.array([df.values for df in self.imputed_datasets])
        mean_imputed = np.mean(imputed_array, axis=0)
        lower_bounds = np.percentile(imputed_array, 2.5, axis=0)
        upper_bounds = np.percentile(imputed_array, 97.5, axis=0)

        missing_mask = self.data.isnull()
        for col_idx, column in enumerate(self.data.columns):
            if missing_mask[column].any():
                missing_rows = missing_mask[column]
                true_vals = self.original_data.loc[missing_rows, column].values
                imputed_vals = mean_imputed[missing_rows.values, col_idx]
                lower = lower_bounds[missing_rows.values, col_idx]
                upper = upper_bounds[missing_rows.values, col_idx]

                rmse = np.sqrt(np.mean((imputed_vals - true_vals) ** 2))
                coverage = np.mean((true_vals >= lower) & (true_vals <= upper))
                avg_interval_width = np.mean(upper - lower)

                results[column] = {
                    "RMSE": rmse,
                    "Coverage_95%": coverage,
                    "Avg_Credible_Interval_Width": avg_interval_width
                }

        print("[Overfitting Evaluation]")
        for col, metrics in results.items():
            print(f" - {col}: RMSE={metrics['RMSE']:.4f}, Coverage={metrics['Coverage_95%']:.2%}, "
                f"Width={metrics['Avg_Credible_Interval_Width']:.4f}")
        
        return results

    def set_priors(self, custom_priors):
        """
        Update the prior distributions used in the Bayesian model.

        Parameters:
        - custom_priors: dict
            A dictionary mapping variable names (columns) to their prior specifications.
            Each value can be a distribution object, a dictionary of parameters,
            or a callable function that returns a sample.

        Example:
        >>> imputer.set_priors({
        ...     'age': {'dist': 'normal', 'mean': 30, 'std': 10},
        ...     'income': lambda: np.random.gamma(2, 1000)
        ... })

        Raises:
        - ValueError: if custom_priors is not a dictionary.
        """
        if not isinstance(custom_priors, dict):
            raise ValueError("custom_priors must be a dictionary.")

        for key in custom_priors:
            if key not in self.data.columns:
                raise ValueError(f"Column '{key}' not found in dataset.")

        self.priors.update(custom_priors)
        print(f"[Set Priors] Updated priors for {list(custom_priors.keys())}")

    def causal_impute(self):
        """
        OPTIONAL: Use the imputed dataset to perform simple causal inference.

        This method is a placeholder to demonstrate how Bayesian imputation can
        feed into a causal analysis pipeline. It assumes that the missing data
        has already been imputed using generate_multiple_imputations().

        Recommended usage:
        - Use imputed data as input to causal models (e.g., using PyMC's do() operator)
        - Define a causal graph (e.g., with causalgraphicalmodels or DoWhy)
        - Perform interventional analysis post-imputation

        Notes:
        - This method does not implement causal inference directly.
        - It serves as a guide for how to use imputed datasets in causal settings.

        References:
        - https://www.pymc.io/projects/examples/en/latest/causal_inference/interventional_distribution.html
        - https://www.pymc.io/projects/examples/en/latest/causal_inference/difference_in_differences.html
        - https://www.pymc-labs.com/blog-posts/causal-analysis-with-pymc-answering-what-if-with-the-new-do-operator/

        Example (pseudo-code):
        >>> imputed_data = imputer.imputed_datasets[0]
        >>> with pm.Model():
        >>>     # Define prior and likelihood with do(x=...)
        >>>     ...
        >>>     trace = pm.sample()

        Raises:
        - NotImplementedError: Placeholder to indicate this is an exploratory extension.
        """
        raise NotImplementedError("Causal inference support is not implemented. See docstring for suggested directions.")

        
