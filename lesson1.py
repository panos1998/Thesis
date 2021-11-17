from collections import Counter

users = [{ "id": 0,  "name":"Hero" },
         { "id": 1,  "name": "Dunn"},
         { "id": 2,  "name": "Sue"},
         { "id": 3,  "name": "Chi"},
         { "id": 4,  "name": "Thor"},
         { "id": 5,  "name": "Clive"},
         { "id": 6,  "name": "Hicks"},
         { "id": 7,  "name": "Devin"},
         { "id": 8,  "name": "Kate"},
         { "id": 9,  "name": "Klein"}
         ]

friendship_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

friendships = {user["id"]: [] for user in users}
print(friendships)
for i, j in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)

print(friendships)

def number_of_friends(user):
    user_id = user["id"]
    friends_ids = friendships[user_id]
    return len(friends_ids)


total_connections = sum(number_of_friends(user)for user in users)
avg = total_connections/len(users)
print(avg)


num_friends_by_id = [(user["id"], number_of_friends(user)) for user in users]
print(num_friends_by_id)
num_friends_by_id.sort(key=lambda id_and_friends: id_and_friends[1], reverse=True)
print(num_friends_by_id)

def foaf(user):
    print([foaf_id
           for friends in friendships[user["id"]]
           for foaf_id in friendships[friends]])
foaf(users[0])

def friends_of_friends(user):
    user_id = user["id"]
    return Counter(
        foaf_id
        for friends_id in friendships[user["id"]]
        for foaf_id in friendships[friends_id]
        if foaf != user_id
        and foaf not in friendships[user_id]
    )

print(friends_of_friends(users[3]))





interests =[(0,"Hadoop"),(0,"Big Data"),(0,"HBase"),(0,"Java"),(0,"Spark"),(0,"Storm"),(0,"Cassandra"),(1,"NoSQL"),
            (1,"MongoDB"),(1,"Cassandra"),(1,"HBase"),(1,"Postgres"),(2,"Python"),(2,"scikit-learn"),(2,"scipy"),(2,"numpy"),
            (2,"statsmodels"),(2,"pandas"),(3,"R"),(3,"Python"),(3,"statistics"),(3,"regression"),(3,"probability"),
            (4,"machine learning"),(4,"regression"),(4,"decision trees"),(4,"libsvm"),(5,"Python"),(5,"R"),(5,"Java"),
            (5,"C++"),(5,"Haskell"),(5,"programming languages"),(6,"statistics"),(6,"probability"),(6,"mathematics"),
            (6,"theory"),(7,"machine learning"),(7,"scikit-learn"),(7,"Mahoot"),(7,"neural networks"),(8,"neural networks"),
            (8,"deep learning"),(8,"Big Data"),(8,"artificial intelligence"),(9,"Hadoop"),(9,"Java"),(9,"MapReduce"),
            (9,"Big Data")]


interests_by_id = {interest[1]: [] for interest in interests}
for interest in interests:
    interests_by_id[interest[1]].append(interest[0])


print(interests_by_id)

def data_scientists_who_like(target):
   return [user_id
           for user_id, interest in interests
           if interest == target]

print(data_scientists_who_like('Big Data'))

# dictionary using defaultdict

from collections import defaultdict


user_ids_by_interest =defaultdict(list)
for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)
print(user_ids_by_interest)

interests_by_user_id = defaultdict(list)
for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)
print(interests_by_user_id)


def most_common_interests(user):
    userid = user["id"]
    print(Counter(
        [common_user_ids
         for interest in interests_by_user_id[userid]
         for common_user_ids in user_ids_by_interest[interest]
         if common_user_ids != userid
         ]
    ))
most_common_interests(users[0])


salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
                        (48000, 0.7), (76000, 6),
                        (69000, 6.5), (76000, 7.5),
                        (60000, 2.5), (83000, 10),
                        (48000, 1.9), (63000, 4.2)]

salary_by_tenure = defaultdict(list)
for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)
print(salary_by_tenure)


average_salary_by_tenure = {
    tenure: sum(salaries)/len(salaries)
    for tenure, salaries in salary_by_tenure.items()
}
print(average_salary_by_tenure)


print(salaries_and_tenures)
def tenure_bucket(tenure):
    if tenure <2:
        return 'less than two'
    elif tenure <5:
        return 'between two and five'
    else:
        return 'more than five'


salary_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)
print(salary_by_tenure_bucket)
average_salary_by_bucket = {
    tenure_bucket: sum(salaries)/len(salaries)
    for tenure_bucket, salaries in salary_by_tenure_bucket.items()
}
print(average_salary_by_bucket)
list = [(0.7,'paid'), (1.9,'unpaid'),(2.5,'paid'),(4.2,'unpaid'),(6.0,'unpaid'),(6.5,'unpaid'),
        (7.5,'unpaid'),(8.1,'unpaid'),(8.7,'paid'),(10.0,'paid')]

def predict_paid_or_unpaid(years_experience):
    if years_experience <3.0:
        return 'paid'
    elif years_experience <8.5:
        return  'unpaid'
    else:
        return 'paid'
prediction_list = []
for years, salary in list:
    prediction = predict_paid_or_unpaid(years)
    prediction_list.append([years,prediction])
print(prediction_list)


words_and_counts = Counter (
    word for user_id, interest in interests
    for word in interest.lower().split()
)
print(words_and_counts)
for word, count in words_and_counts.most_common():
    if count >1:
        print(word, count)







