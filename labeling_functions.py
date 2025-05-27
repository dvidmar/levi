"""Module for Labeling Functions."""
import re
import numpy as np
import pandas as pd

from typing import List


class RegexVoter:
    """Runs regex across data to determine votes."""
    def __init__(
        self,
        pattern: str,
        apply_to: str,
        negate: bool = False
    ):
        """
        Args:
            pattern (RegexPattern): Regex pattern to use.
            apply_to (str): Dataframe field to run regex on.
            negate (bool, optional): Whether to vote if pattern is negated.
                Defaults to False.
        """
        self.pattern = pattern
        self.apply_to = apply_to
        self.negate = negate

    def vote(self, df:pd.DataFrame) -> pd.Series:
        """Vote on all data using regex contains logic.

        Args:
            df (pd.DataFrame): Dataset to vote on.

        Returns:
            pd.Series: Boolean series, with votes as `True`.
        """

        if self.negate:
            return ~(
                df[self.apply_to].str.contains(self.pattern, flags=re.I)
            )
        else:
            return df[self.apply_to].str.contains(self.pattern, flags=re.I)


class LabelingFunction:
    """Base class for labeling functions."""
    def __init__(self, 
                voters: List[RegexVoter], 
                name: str, 
                label: int):
        """
        Args:
            voters (List[RegexVoter]): Voters to combine via "AND" logic.
            name (str): Name of LF.
            label (int): Label to be given to LF votes, e.g. does this LF
                vote for positives or negatives.
        """
        self.voters = voters
        self.label = label
        self.name = name

        # store dict view for yaml building
        self.dict_view = {
                'name': self.name,
                'label': self.label,
                'voters': [
                    {
                        'regex': voter.pattern,
                        'apply_to': voter.apply_to,
                        'negate': voter.negate
                    }
                    for voter in self.voters
                ]
            }

    def vote(self, df:pd.DataFrame) -> list:
        """Use all voters to pull LF cohort across dataset.

        Args:
            df (pd.DataFrame): Dataset in dataframe view.

        Returns:
            list: Indices of LF cohort (samples who this LF votes for) across dataset.
        """
        votes = [voter.vote(df) for voter in self.voters]
        return list(
            df[np.logical_and.reduce(votes)].index
        )
