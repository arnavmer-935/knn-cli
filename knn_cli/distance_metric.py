from math import sqrt

def euclidean(p: list[float], q: list[float]) -> float:
    distance_squared = 0
    pi, qi = 0, 0
    while pi < len(p) and qi < len(q):
        distance_squared += (p[pi] - q[qi]) ** 2
        pi += 1
        qi += 1

    return sqrt(distance_squared)

def manhattan(p: list[float], q: list[float]) -> float:
    m_distance = 0
    pi, qi = 0, 0
    while pi < len(p) and qi < len(q):
        m_distance += abs(p[pi] - q[qi])
        pi += 1
        qi += 1
    return m_distance

def cosine(p: list[float], q: list[float]) -> float:
    pi, qi, dot_product = 0, 0, 0
    while pi < len(p) and qi < len(q):
        dot_product += (p[pi] * q[qi])
        pi += 1
        qi += 1

    p_length = sqrt(sum(x ** 2 for x in p))
    q_length = sqrt(sum(x ** 2 for x in q))

    similarity = 1 - (dot_product / (p_length * q_length))
    return similarity