from math import sqrt

def euclidean(p: list[float], q: list[float]) -> float:
    """
    Calculates the Euclidean (straight-line) distance between two vectors.

    Raises ValueError if the vectors have different lengths or are empty.

    :param p: first vector as a list of floats.
    :param q: second vector as a list of floats.

    :return: Euclidean distance between p and q.
    """
    if len(p) != len(q):
        raise ValueError("Vectors must have equal lengths.")

    if not p:
        raise ValueError("Vectors must have at least 1 dimension.")

    distance_squared = 0
    for i in range(len(p)):
        distance_squared += (p[i] - q[i]) ** 2

    return sqrt(distance_squared)

def manhattan(p: list[float], q: list[float]) -> float:
    """
    Calculates the Manhattan (city block) distance between two vectors.

    Raises ValueError if the vectors have different lengths or are empty.

    :param p: first vector as a list of floats.
    :param q: second vector as a list of floats.

    :return: Manhattan distance between p and q.
    """
    if len(p) != len(q):
        raise ValueError("Vectors must have equal lengths.")

    if not p:
        raise ValueError("Vectors must have at least 1 dimension.")

    m_distance = 0
    for i in range(len(p)):
        m_distance += abs(p[i] - q[i])

    return m_distance

def cosine(p: list[float], q: list[float]) -> float:
    """
    Calculates the cosine distance between two vectors, defined as 1 minus
    the cosine similarity. A value of 0 means the vectors point in the same
    direction, and 2 means they point in opposite directions.

    Raises ValueError if the vectors have different lengths, are empty,
    or if either vector is a zero vector.

    :param p: first vector as a list of floats.
    :param q: second vector as a list of floats.

    :return: cosine distance between p and q in the range [0, 2].
    """
    if len(p) != len(q):
        raise ValueError("Vectors must have equal lengths.")

    if not p:
        raise ValueError("Vectors must have at least 1 dimension.")

    dot_product = 0
    for i in range(len(p)):
        dot_product += (p[i] * q[i])

    p_length = sqrt(sum(x ** 2 for x in p))
    q_length = sqrt(sum(x ** 2 for x in q))

    if p_length == 0 or q_length == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    return 1 - (dot_product / (p_length * q_length))