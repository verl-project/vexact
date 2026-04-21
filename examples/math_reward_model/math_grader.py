# Copyright (c) [year] sail-sg/Precision-RL-verl
# Source: https://github.com/sail-sg/Precision-RL-verl


import multiprocessing


def compute_math_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    # Multi-turn tool-calling path: reward comes from MathTool via tool_rewards
    if extra_info:
        tool_rewards = extra_info.get("tool_rewards", [])
        if tool_rewards:
            score = max(tool_rewards)
            return {"score": score, "formatted": True}
    THINK_END = "</think>"
    if solution_str.count(THINK_END) != 1:
        return {
            "score": 0.0,
            "formatted": False,
            # "sympy": False,
            # "invalid_gt": False,
        }
    solution_str = solution_str.split(THINK_END)[1]

    from experimental.math_reward_model.math_utils import extract_answer, grade_answer_mathd

    model_answer = extract_answer(solution_str)
    if model_answer is None:
        return {
            "score": 0.0,
            "formatted": False,
            # "sympy": False,
            # "invalid_gt": False,
        }

    ground_truths = ground_truth
    # Process the ground truth(s)
    if ground_truths is None:
        print(f"ground_truths is None: {ground_truths}")
        return {
            "score": 0.0,
            "formatted": False,
            # "sympy": False,
            # "invalid_gt": True,
        }

    # Convert single answer to list for uniform processing
    if isinstance(ground_truths, (str, float, int)):
        ground_truths = [ground_truths]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        print(f"processed_ground_truths is empty: {ground_truths}")
        return {
            "score": 0.0,
            "formatted": False,
            # "sympy": False,
            # "invalid_gt": True,
        }

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth)
        if is_correct:
            return {
                "score": 1.0,
                "formatted": True,
                # "sympy": False,
                # "invalid_gt": False,
            }
        is_correct = run_grade_answer_sympy_with_timeout(model_answer, ground_truth, 10.0)
        if is_correct:
            return {
                "score": 1.0,
                "formatted": True,
                # "sympy": True,
                # "invalid_gt": False,
            }

    return {
        "score": 0.0,
        "formatted": True,
        # "sympy": False,
        # "invalid_gt": False,
    }


class ProcessPoolManager:
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pool = None
        return cls._instance

    def get_pool(self, processes=None):
        if self._pool is None:
            self._pool = multiprocessing.Pool(processes=processes)
        return self._pool

    def terminate(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __del__(self):
        self.terminate()


def run_grade_answer_sympy_with_timeout(model_answer: str, ground_truth: str, timeout: float = 10.0) -> bool:
    from experimental.math_reward_model.math_utils import grade_answer_sympy

    process_pool_manager = ProcessPoolManager()
    pool = process_pool_manager.get_pool()
    result = pool.apply_async(grade_answer_sympy, (model_answer, ground_truth))
    try:
        return result.get(timeout)
    except multiprocessing.TimeoutError:
        print(f"sympy timeout, model_answer: {model_answer}, ground_truth: {ground_truth}")
        return False
