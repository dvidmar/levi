from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
import yaml

from io import BytesIO
from typing import Tuple

import scipy
from scipy.special import betainc, digamma, beta

from labeling_functions import LabelingFunction, RegexVoter

def split_headers_body(text: str) -> Tuple[str,str]:
    """Splits a newsgroup message into headers and body.

    Args:
        text (str): Message text.

    Returns:
        Tuple[str,str]: Header and body text.
    """

    parts = text.split('\n\n', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        return parts[0], ''

def load_toy_data() -> pd.DataFrame:
    """Load toy dataset from sklearn (20 newsgroups) and split into various `modalities`.

    Returns:
        pd.DataFrame: Processed 20 newsgroup data sample.
    """
    # (1) load toy data
    newsgroups_data = fetch_20newsgroups(subset='train')

    # (2) split out headers from body
    headers = []
    bodies = []
    for text in newsgroups_data.data:
        header, body = split_headers_body(text)
        headers.append(header)
        bodies.append(body)

    # (3) gather data, with different "modalities" as we would find in EHR-type data
    return pd.DataFrame({
        'text':  bodies,
        'header': headers,
        'category': [newsgroups_data.target_names[target_idx] for target_idx in newsgroups_data.target]
    })

def transform_vote_domain(votes: pd.DataFrame) -> Tuple[list,np.array]:
    """Transform votes values from 0,1 -> -1,1 (if necessary).

    Args:
        votes (pd.DataFrame): Votes dataframe

    Raises:
        ValueError: Votes must either be 0,1 or -1,1

    Returns:
        Tuple[list,np.array]: LF names and votes values.
    """
    # conform to (-1,1) votes domain
    lf_names = list(votes)
    vote_domain = set(np.unique(votes).astype(int))
    
    if vote_domain.issubset(set([0,1])):
        votes_ = 2 * votes.values - 1
    elif vote_domain.issubset(set([-1,1])):
        votes_ = votes.values
    else:
        raise ValueError(f'Votes must either be 0,1 or -1,1.  Instead got {set(votes)}')
    
    return lf_names, votes_

def find_maxent_ab(max_plausible_value: float, max_percentile: float = 0.95) -> Tuple[float, float]:
    """Find the maximum entropy beta distribution with the given percentile max value.

    Args:
        max_plausible_value (float): Value which we want distribution percentile contained below.
        max_percentile (float): Percentile to be contained below value, defaults to 0.95

    Returns:
        Tuple[float,float]: Alpha and beta parameters for beta distribution.
    """
    def func(ab):
        return -(
            np.log(beta(ab[0], ab[1])) -
            (ab[0] - 1) * digamma(ab[0]) - (ab[1] - 1) * digamma(ab[1]) +
            (ab[0] + ab[1] - 2) * digamma(ab[0] + ab[1])
        ) # this is the (negative) expression for entropy of a beta distribution (see https://en.wikipedia.org/wiki/Beta_distribution)

    def con(ab):
        return betainc(ab[0], ab[1], max_plausible_value) - max_percentile

    bnds = ((1e-10, None), (1e-10, None))  # ensure alpha, beta > 0
    res = scipy.optimize.minimize(func, (1,1), constraints=[{'type':'eq', 'fun': con}], bounds=bnds)  # minimize negative of entropy, eg maximize entropy

    ab = res['x']
    return ab[0], ab[1]

def load_lf_yaml(fname: str) -> dict:
    """Load LFs YAML file.

    Args:
        fname (str): Filename.

    Returns:
        dict: User defined LFs.
    """
    yaml_str = BytesIO(open(fname, "rb").read())
    return yaml.safe_load(yaml_str)

def lf_from_dict(lf: dict) -> LabelingFunction:
    """Create LabelingFunction object from dict of params.

    Args:
        lf (dict): Dictionary of params for this LF.

    Returns:
        LabelingFunction: LF object with given params.
    """
    return LabelingFunction(
            [
                RegexVoter(
                    voter['regex'],
                    apply_to = voter['apply_to'],
                    negate = voter['negate']
                )
                for voter in lf['voters']
            ],
            name = lf['name'],
            label = lf['label']
        )

def get_lf_polarities(LFs: dict) -> dict:
    """Gather which label each LF is voting for (positive or negative)

    Args:
        LFs (dict): LF definitions.

    Returns:
        dict: LF polarities.
    """
    return {
        LF['name']: LF['label']
        for LF in LFs
    }