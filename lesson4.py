from typing import List
from collections import Counter
import matplotlib.pyplot as plt
from lesson3 import sum_of_squares
num_friends = [100.0,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


friends_counts = Counter(num_friends)
xs = range(101)
ys = [friends_counts[x] for x in xs]
plt.bar(xs, ys)
plt.axis([0, 101, 0, 25])
plt.title("Ιστόγραμμα αριθμού φίλων")
plt.xlabel('# φίλων')
plt.ylabel('# μελών')
plt.show()
num_points = len(num_friends)
largest_value = max(num_friends)
smallest_value = min(num_friends)
sorted_values = sorted(num_friends)
smallest_value = sorted_values[0]
second_smallest_value = sorted_values[1]
second_larges_value = sorted_values[-2]


def mean(xs: List[float]) -> float:
    return sum(xs)/len(xs)


mean(num_friends)


def _median_odd(xs: List[float]) -> float:
    return sorted(xs)[len(xs)//2]


def _median_even(xs: List[float]) -> float:
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2
    return (sorted_xs[hi_midpoint-1]+sorted_xs[hi_midpoint]) / 2


def median(v: List[float]) -> float:
    return _median_even(v) if len(v) % 2 ==0 else _median_odd(v)


assert median([1, 10, 2, 9, 5]) == 5
assert median([1, 9, 2, 10]) == (2 + 9) / 2


print(median(num_friends))


def quantile(xs: List[float], p: float) -> float:
    p_index = int(p*len(xs))
    return sorted(xs)[p_index]


assert quantile(num_friends, 0.1) == 1


def mode(x: List[float]) -> List[float]:
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]


assert set(mode(num_friends)) == {1, 6}


def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)


assert data_range(num_friends) == 99


def de_mean(xs: List[float]) -> List[float]:
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


def variance(xs: List[float]) -> float:
    assert len(xs) >= 2, "η διακύμανση χρειάζεται τουλάχιστον 2 στοιχεία"
    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n-1)


assert 81.54 < variance(num_friends) < 81.55


import math


def standard_deviation(xs: List[float]) -> float:
    return math.sqrt(variance(xs))


assert 9.02 < standard_deviation(num_friends) < 9.04


def interquartile_range(xs: List[float]) -> float:
    return quantile(xs, 0.75) - quantile(xs, 0.25)


assert interquartile_range(num_friends) == 6


from lesson3 import dot


def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), 'oi xs ys prepei na exoun ido plithos stoixion'
    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)


 #assert 22.42 < covariance(num_friends, daily_minutes) < 22.43


def correlation(xs: List[float], ys: List[float]) -> float:
    stdev_x =standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x /stdev_y
    else:
        return 0


#assert 0.24 <correlation(num_friends, daily_minutes) < 0.25
#assert 0.24 < correlation(num_friends, daily_hours) < 0.25



