import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def soft_rank(scores: np.ndarray, tau: float = 0.05) -> np.ndarray:
    """
    Differentiable approximation of ranks.
    Return: soft ranks where 1 is best (larger score -> smaller rank).
    Complexity O(n^2) per call; tau controls smoothness (larger tau -> smoother).
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    r = np.ones(n)
    for i in range(n):
        diff = (scores - scores[i]) / tau
        r[i] += np.sum(sigmoid(diff)) - sigmoid(0.0)  # remove j=i term
    return r

def rule_type_for_season(season: int) -> str:
    # As described in the problem appendix:
    # rank: seasons 1-2, and reasonably assume 28-34
    # percent: seasons 3-27
    if season in (1, 2):
        return "S1-2_rank"
    if 3 <= season <= 27:
        return "S3-27_percent"
    return "S28+_bottom2"
