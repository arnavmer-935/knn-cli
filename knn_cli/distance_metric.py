from math import sqrt

def euclidean(p: list[float], q: list[float]) -> float:
    if len(p) != len(q):
        raise ValueError("Vectors must have equal lengths.")

    if not p:
        raise ValueError("Vectors must have at least 1 dimension.")

    distance_squared = 0
    for i in range(len(p)):
        distance_squared += (p[i] - q[i]) ** 2

    return sqrt(distance_squared)

def manhattan(p: list[float], q: list[float]) -> float:
    if len(p) != len(q):
        raise ValueError("Vectors must have equal lengths.")

    if not p:
        raise ValueError("Vectors must have at least 1 dimension.")

    m_distance = 0
    for i in range(len(p)):
        m_distance += abs(p[i] - q[i])

    return m_distance

def cosine(p: list[float], q: list[float]) -> float:
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