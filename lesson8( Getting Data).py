import re
from collections import Counter

#file_for_reading = open('reading_file.txt', 'r')
#file_for_reading_2 = open('reading_file.txt')
#file_for_writing = open('writing_file.txt', 'w')
#file_for_appending = open('appending_file.txt', 'a')
#file_for_writing.close()

#with open(filename) as f:
#    data = function_that_gets_data_from(f)
#process(data)

starts_with_has = 0
#with open('input.txt') as f:
 #   for line in f:
  #      if re.match("^#", line):
   #         starts_with_has += 1


def get_domain(email_adress: str):
    return email_adress.lower().split('@')[-1]


assert get_domain('joelgrus@gmail.com') == 'gmail.com'
assert get_domain('joel@m.datasciencester.com') == 'm.datasciencester.com'

with open('email_adresses.txt', 'r') as f:
    domain_counts = Counter(get_domain(line.strip())
                            for line in f
                            if '@' in line)
    print(domain_counts)


import csv

with open('tab_delimited_stock_prices.txt') as f:
    tab_reader = csv.reader(f, delimiter='\t')
    for row in tab_reader:
        date = row[0]
        #symbol = row[1]
        #closing_price = float(row[2])
        print(date)