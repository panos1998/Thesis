from typing import List, Dict
from collections import Counter
import random
from lesson5 import inverse_normal_cdf
import math
import matplotlib.pyplot as plt
from lesson4 import correlation
from lesson3 import Matrix, Vector, make_matrix

# Αυτή είναι η μέθοδος για να ομαδοποιούμε στοιχεία σε κάδους ανάλογα το μέγεθος  της τιμής τους
# και ανάλογα  το εύρος  τιμών(πλατος)  κάθε κάδου bucket_size

def bucketize(point: float, bucket_size: float) -> float:
    return bucket_size * math.floor(point / bucket_size)


def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    return Counter(bucketize(point, bucket_size) for point in points)


def plot_histogram(points: List[float], bucket_size: float, title: str = ""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()


random.seed(0)
uniform = [200 * random.random() - 100 for _ in range(10000)]
normal = [57 * inverse_normal_cdf(random.random())
          for _ in range(10000)]
plot_histogram(uniform, 10, "Το ιστόγραμμα του συνόλου uniform")
plot_histogram(normal, 10, "Το ιστόγραμμα του συνόλου  Normal")

"""Επιστρέφει ενα τυχαιο δειγμα απο μια τυπικη κανονικη κατανομη"""


def random_normal() -> float:
    return inverse_normal_cdf(random.random())


xs = [random_normal() for _ in range(1000)]
ys1 = [x + random_normal() / 2 for x in xs]
ys2 = [-x + random_normal() / 2 for x in xs]
plot_histogram(ys1, 10, 'plot ys1')
plot_histogram(ys2, 10, 'plot ys2')

"""Διάγραμμα διασποράς"""


plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Πολυ διαφορετικές απο κοινού κατανομές")
plt.show()
print(correlation(xs, ys1))
print(correlation(xs, ys2))


def correlation_matrix(data: List[Vector]) -> Matrix:
    def correlation_ij(i: int, j: int) -> float:
        return correlation(data[i], data[j])
    return make_matrix(len(data), len(data), correlation_ij)


assert 0.89 < correlation(xs, ys1) < 0.91
assert -0.91 < correlation(xs, ys2) < -0.89

corr_data = [[math.floor(inverse_normal_cdf(random.random())) for _ in range(6000)] for _ in range(4)]
num_vectors = len(corr_data)
fig, ax = plt.subplots(num_vectors, num_vectors)
for i in range(num_vectors):
    for j in range(num_vectors):
        if i != j: ax[i][j].scatter(corr_data[j], corr_data[i])
        else: ax[i][j].annotate("series" + str(i), (0.5, 0.5),
                                xycoords='axes fraction',
                                ha='center', va='center')
        if i < num_vectors -1: ax[i][j].xaxis.set_visible(False)
        if j > 0: ax[i][j].yaxis.set_visible(False)
ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_ylim())
plt.show()


#NamedTuples
import datetime
stock_price ={'closing_price': 102.06,
              'date': datetime.date(2014, 8, 29),
              'symbol': 'AAPL'}
prices: Dict[datetime.date, float] = {}
from  collections import namedtuple
StockPrice = namedtuple('StockPrice',['symbol', 'date', 'closing_price'])
price = StockPrice('MSFT',datetime.date(2018, 12, 14), 106.03)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.03

from typing import NamedTuple


class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']


price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)


assert price.symbol == 'MSFT'
assert price.closing_price == 106.03
assert price.is_high_tech()

from dataclasses import dataclass

@dataclass
class StockPrice2:
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']


price2 = StockPrice2('MSFT', datetime.date(2018, 12, 14), 106.03)


assert price2.symbol == 'MSFT'
assert price2.closing_price == 106.03
assert price2.is_high_tech()

price2.closing_price /= 2
assert price2.closing_price == 53.015

from dateutil.parser import parse


def parse_row(row: List[str]) -> StockPrice:
    symbol, date, closing_price = row
    return StockPrice(symbol=symbol,
                      date=parse(date).date(),
                      closing_price=float(closing_price))


stock = parse_row(['MSFT', '2018-12-14', '106.03'])

from typing import Optional
import  re


def try_parse_row(row: List[str]) -> Optional[StockPrice]:
    symbol, date, closing_price = row
    if not re.match(r"^[A-Z]+$", symbol):
        return  None

    try:
        date = parse(date).date()
    except ValueError:
        return None
    try:
        closing_price = float(closing_price)
    except ValueError:
        return None

    return StockPrice(symbol, date, closing_price)

# Should return None for errors


assert try_parse_row(["MSFT0", "2018-12-14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12--14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12-14", "x"]) is None

# But should return same as before if data is good.
assert try_parse_row(["MSFT", "2018-12-14", "106.03"]) == stock

import csv

data: List[StockPrice] = []
with open('file.txt') as f:
    reader = csv.reader(f)
    for row in reader:
        maybe_stock = try_parse_row(row)
        if maybe_stock is None:
            print(f'παράλειψη ακυρης γραμμης: {row}')
        else:
            data.append(maybe_stock)
            print(maybe_stock)
