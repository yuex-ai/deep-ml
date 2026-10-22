import torch
import math
from collections import Counter
from typing import List, Dict, Any, Union


def calculate_entropy(labels: List[Any]) -> float:
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return 0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def calculate_information_gain(
    examples: List[Dict[str, Any]],
    attr: str,
    target_attr: str
) -> float:
    total_labels = [ex[target_attr] for ex in examples]
    total_ent = calculate_entropy(total_labels)
    n = len(examples)
    rem = 0.0
    for v in set(ex[attr] for ex in examples):
        subset_labels = [ex[target_attr] for ex in examples if ex[attr] == v]
        rem += (len(subset_labels)/n) * calculate_entropy(subset_labels)
    return total_ent - rem


def majority_class(
    examples: List[Dict[str, Any]],
    target_attr: str
) -> Any:
    counts = Counter(ex[target_attr] for ex in examples)
    max_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == max_count]
    return sorted(candidates)[0]


def learn_decision_tree(
    examples: List[Dict[str, Any]],
    attributes: List[str],
    target_attr: str
) -> Union[Dict[str, Any], Any]:
    if not examples:
        return 'No examples'
    first_label = examples[0][target_attr]
    if all(ex[target_attr] == first_label for ex in examples):
        return first_label
    if not attributes:
        return majority_class(examples, target_attr)
    gains = [(a, calculate_information_gain(examples, a, target_attr)) for a in attributes]
    best = max(gains, key=lambda x: x[1])[0]
    tree: Dict[str, Any] = {best: {}}
    for v in sorted(set(ex[best] for ex in examples)):
        subset = [ex for ex in examples if ex[best] == v]
        rem_attrs = [a for a in attributes if a != best]
        tree[best][v] = learn_decision_tree(subset, rem_attrs, target_attr)
    return tree