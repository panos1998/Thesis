from typing import List
import math


def entropy(class_probabilities: List[float]) -> float:
    return sum(-p * math.log(p, 2) for p in class_probabilities
               if p > 0)


assert entropy([1.0]) == 0
assert entropy([0.5, 0.5]) == 1
assert 0.81 < entropy([0.25, 0.75]) < 0.82

from typing import Any
from collections import Counter


def class_probabilities(labels: List[Any]) -> List[float]:
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]


def data_entropy(labels: List[Any]) -> float:
    return entropy(class_probabilities(labels))


assert data_entropy(['a']) == 0
assert data_entropy([True, False]) == 1
assert data_entropy([3, 4, 4, 4]) == entropy([0.25, 0.75])


def partition_entropy(subsets: List[List[Any]]) -> float:
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count
               for subset in subsets)


from typing import NamedTuple, Optional


class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None


inputs = [
Candidate('Senior', 'Java', False, False, False),
Candidate('Senior', 'Java', False, True, False),
Candidate('Mid', 'Python', False, False, True),
Candidate('Junior', 'Python', False, False, True),
Candidate('Junior', 'R', True, False, True),
Candidate('Junior', 'R', True, True, False),
Candidate('Mid', 'R', True, True, True),
Candidate('Senior', 'Python', False, False, False),
Candidate('Senior', 'R', True, False, True),
Candidate('Junior', 'Python', True, False, True),
Candidate('Senior', 'Python', True, True, True),
Candidate('Mid', 'Python', False, True, True),
Candidate('Mid', 'Java', True, False, True),
Candidate('Junior', 'Python', False, True, False)
]

from typing import Dict, TypeVar
from collections import defaultdict

T = TypeVar('T')


def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    partitions: Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute)
        partitions[key].append(input)
    return partitions


def partition_entropy_by(inputs: List[Any],
                         attribute: str,
                         label_attribute: str) -> float:
    partitions = partition_by(inputs, attribute)
    labels = [[getattr(input, label_attribute) for input in partition]
              for partition in partitions.values()]
    return partition_entropy(labels)


for key in ['level', 'lang', 'tweets', 'phd']:
    print(key, partition_entropy_by(inputs, key, 'did_well'))

senior_inputs = [input for input in inputs if input.level == 'Senior']

from typing import Union
class Leaf(NamedTuple):
    value: Any


class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None


DecisionTree = Union[Leaf, Split]


def classify(tree: DecisionTree, input: Any) -> Any:
    if isinstance(tree, Leaf):
        return tree.value
    subtree_key = getattr(input, tree.attribute)
    if subtree_key not in tree.subtrees:
        return tree.default_value
    subtree = tree.subtrees[subtree_key]
    return classify(subtree, input)


def build_tree_id3(inputs: List[Any],
                   split_attributes: List[str],
                   target_attribute: str) -> DecisionTree:
    label_counts = Counter(getattr(input, target_attribute)
                           for input in inputs)
    most_common_label = label_counts.most_common(1)[0][0]
    if len(label_counts) == 1:
        return Leaf(most_common_label)
    if not split_attributes:
        return Leaf(most_common_label)

    def split_entropy(attribute: str) -> float:
        return partition_entropy_by(inputs, attribute, target_attribute)
    best_attribute = min(split_attributes, key=split_entropy)
    partitions = partition_by(input, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]
    subtress = {attribute_value: build_tree_id3(subset,
                                                new_attributes,
                                                target_attribute)
                for attribute_value, subset in partitions.items()}
    return Split(best_attribute, subtress, default_value=most_common_label)
