from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
import itertools

import numpy as np
import tqdm

from .execution import check_correctness


def estimate_pass_at_k(num_samples: Union[int, List[int], np.ndarray],
                       num_correct: Union[List[int],
                                          np.ndarray], k: int) -> np.ndarray:
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)"""
        if n - c < k:
            # if this were the case, we would have succeeded regardless of the
            # order of the samples
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    results = []
    for n, c in zip(num_samples_it, num_correct):
        results.append(estimator(n, c, k))
    return np.array(results)


def evaluate_functional_correctness(
    problems,
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """Evaluates the functional correctness of generated samples."""
    # check the generated samples against the test suites
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        # index the problems by result_id so we can update the dictionaries later
        # regardless of completion order
        results = {p['completion_id']: p for p in problems}
        for problem in problems:
            completion_id = problem['completion_id']
            args = (problem, timeout, completion_id)
            future = executor.submit(check_correctness, *args)
            futures.append(future)

        print("Running test suite...")
        for future in tqdm.tqdm(as_completed(futures), total=len(problems)):
            result = future.result()
            # update the results dictionary
            results[result['completion_id']].update(result)
    # convert back to list
    results = list(results.values())
    return results


def calculate_pass_at_k(generations: List[Dict],
                        k: List[int] = [1, 10]):
    """Assumes that generations are grouped by a particular hypothesis. Within
    each hypothesis, for each task_id, we must calculate correct and total.

    This function will need to be called once per hypothesis.
    """
    correct = Counter()
    total = Counter()
    for generation in generations:
        task_id = generation['task_id']
        passed = generation['passed']
        correct[task_id] += int(passed)
        total[task_id] += 1

    # convert to numpy arrays
    correct = np.array(list(correct.values()))
    total = np.array(list(total.values()))

    # calculate pass@k. NB: only calculate for k where we have at least k samples
    ks = k
    pass_at_k = {
        f'pass@{k}': estimate_pass_at_k(total, correct, k).mean()
        for k in ks if (total >= k).all()
    }
    return pass_at_k
