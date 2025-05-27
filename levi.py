import numpy as np
import pandas as pd

from utils import transform_vote_domain, get_lf_polarities
from labeling_functions import LabelingFunction

class VotesData:
    def __init__(self, data: pd.DataFrame):
        """Data holding Labeling Function votes across data.

        Args:
            data (pd.DataFrame): Data.
        """
        self.data = data

        if not self.data.index.is_unique:
            raise ValueError('Data index values must be unique')

        self.df = pd.DataFrame(index=self.data.index)  # rows = samples, columns = LFs

    def apply(self, lf: LabelingFunction):
        """Apply a Labeling Function to vote on data.

        Args:
            lf (LabelingFunction): Labeling Function object.
        """
        # get list of sample_ids with LF votes
        cohort = lf.vote(
            self.data
        )

        # convert to df view
        self.df.loc[:, lf.name] = 0
        self.df.loc[cohort, lf.name] = 1

class LEVI:
    def __init__(self, LFs: dict, ab_priors: dict):
        """Class implementing LEVI aggregation on votes data.

        Args:
            LFs (dict): LFs loaded from YAML into dict.
            ab_priors (dict): Definition of relevant priors.
        """
        # set polarities
        self.lf_polarity = get_lf_polarities(LFs)

        # set priors
        self._set_prior_ab(ab_priors)

    def _set_prior_ab(self, ab_priors: dict):
        """Set beta distribution parameters for priors

        Args:
            ab_priors (dict): Definition of relevant priors.
        """
        self.rho_a = ab_priors['rho_a']
        self.rho_b = ab_priors['rho_b']

        self.s_a = {}
        self.s_b = {}
        self.z_a = {}
        self.z_b = {}

        for lf_name, lf_polarity in self.lf_polarity.items():
            if lf_polarity > 0:  # positive LF
                self.s_a[lf_name] = ab_priors['pos_tpr_a']
                self.s_b[lf_name] = ab_priors['pos_tpr_b']
                self.z_a[lf_name] = ab_priors['pos_fpr_a']
                self.z_b[lf_name] = ab_priors['pos_fpr_b']
            else:  # negative LF (NOTE: negative LFs "flip" definition of tpr <> fpr)
                self.s_a[lf_name] = ab_priors['pos_fpr_a']
                self.s_b[lf_name] = ab_priors['pos_fpr_b']
                self.z_a[lf_name] = ab_priors['pos_tpr_a']
                self.z_b[lf_name] = ab_priors['pos_tpr_b']
    
    def predict_proba(self, votes_df: pd.DataFrame) -> pd.Series:
        """Predict probabilities from votes data using LEVI.

        Args:
            votes_df (pd.DataFrame): Votes data in dataframe view.

        Returns:
            pd.Series: Output probabilities with sample IDs as index.
        """
        # conform to (-1,1) votes domain
        lf_names, votes = transform_vote_domain(votes_df)  # convert to (0,1) -> (-1,1)

        # compute params from priors
        mu_0 = np.log(self.rho_a / self.rho_b) + np.sum(
            [
                np.log(
                    np.sqrt(self.s_a[lf_name] * self.s_b[lf_name] / (self.s_a[lf_name] + self.s_b[lf_name]) ** 2)
                ) - np.log(
                    np.sqrt(self.z_a[lf_name] * self.z_b[lf_name] / (self.z_a[lf_name] + self.z_b[lf_name]) ** 2)
                )
                for lf_name in lf_names
            ]
        )

        mu_i = [
            np.log(
                np.sqrt(self.z_b[lf_name] / self.z_a[lf_name])
            ) - np.log(
                np.sqrt(self.s_b[lf_name] / self.s_a[lf_name])
            )
            for lf_name in lf_names
        ]

        p = np.array([
            (
                1 + np.exp(
                    -mu_0 - np.dot(mu_i, votes[idx])
                )
            ) ** (-1)
            for idx in range(len(votes))
        ])

        return pd.Series(p, index=votes_df.index, name='p')
