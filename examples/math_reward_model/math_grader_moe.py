# Copyright (c) [year] sail-sg/Precision-RL-verl
# Source: https://github.com/sail-sg/Precision-RL-verl
#
# Modified version to support models without DeepSeek-style thought tags (e.g., Qwen)


import multiprocessing


def extract_answer_with_fallback(solution_str: str) -> tuple:
    """
    Extract answer with fallback logic.
    Returns (answer, extraction_source) where extraction_source is one of:
    'boxed' or None if extraction failed.
    """
    from examples.math_reward_model.math_utils import extract_answer

    # Try standard \boxed{} extraction
    model_answer = extract_answer(solution_str)
    if model_answer is not None:
        return model_answer, "boxed"

    return None, None


def compute_math_score(
    data_source,
    solution_str,
    ground_truth,
    **kwargs,
):
    # ===== DEBUG: output the input data =====
    # print("=" * 80)
    # print(f"[REWARD INPUT] data_source: {data_source}")
    # print(f"[REWARD INPUT] ground_truth: {ground_truth}")
    # print(f"[REWARD INPUT] solution_str pre1000:\n{solution_str[:1000]}")
    # print("=" * 80)
    # ===== DEBUG END =====

    from examples.math_reward_model.math_utils import extract_answer, grade_answer_mathd

    # Handle thought tags flexibly - support both DeepSeek-style and Qwen-style outputs
    THINK_START = "<think>"
    THINK_END = "</think>"
    think_end_count = solution_str.count(THINK_END)

    # extraction_context = None

    if think_end_count == 1:
        # DeepSeek-style: extract content after </think>
        answer_region = solution_str.split(THINK_END)[1]
        # extraction_context = "after_think_tag"
        # print("[REWARD DEBUG] Extraction mode: DeepSeek-style (after </think> tag)")
    elif think_end_count == 0:
        # Qwen-style or other models: no thought tags
        if THINK_START in solution_str:
            # Unclosed <think> tag - take content after it
            answer_region = solution_str.split(THINK_START)[-1]
            # extraction_context = "after_unclosed_think"
            # print("[REWARD DEBUG] Extraction mode: Unclosed <think> tag, using content after it")
        else:
            # No thought tags at all - use the entire output
            answer_region = solution_str
            # extraction_context = "full_output"
            # print("[REWARD DEBUG] Extraction mode: No thought tags (Qwen-style), using full output")
    else:
        # Multiple </think> tags - use content after the last one
        answer_region = solution_str.split(THINK_END)[-1]
        # extraction_context = "after_last_think_tag"
        # print(
        # f"[REWARD DEBUG] Extraction mode: Multiple </think> tags ({think_end_count}), using content after last tag"
        # )

    # Extract answer with fallback support
    model_answer, extraction_source = extract_answer_with_fallback(answer_region)

    if model_answer is None:
        print(f"[REWARD FAIL] Reason 2: Cannot extract answer. Content preview: {answer_region[:500]}")
        return {
            "score": 0.0,
            "formatted": False,
        }

    ground_truths = ground_truth
    if ground_truths is None:
        print("[REWARD FAIL] Reason 3: ground_truth is None")
        return {
            "score": 0.0,
            "formatted": False,
        }

    if isinstance(ground_truths, (str, float, int)):
        ground_truths = [ground_truths]

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
        print(f"[REWARD FAIL] Reason 4: processed_ground_truths is empty, original: {ground_truths}")
        return {
            "score": 0.0,
            "formatted": False,
        }

    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth)
        if is_correct:
            return {
                "score": 1.0,
                "formatted": True,
            }
        is_correct = run_grade_answer_sympy_with_timeout(model_answer, ground_truth, 10.0)
        if is_correct:
            return {
                "score": 1.0,
                "formatted": True,
            }

    print(
        f"[REWARD FAIL] Reason 5: Answer mismatch, model_answer: {model_answer}, "
        f"truths: {processed_ground_truths}, extraction: {extraction_source}"
    )
    return {
        "score": 0.0,
        "formatted": True,
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
    from examples.math_reward_model.math_utils import grade_answer_sympy

    process_pool_manager = ProcessPoolManager()
    pool = process_pool_manager.get_pool()
    result = pool.apply_async(grade_answer_sympy, (model_answer, ground_truth))
    try:
        return result.get(timeout)
    except multiprocessing.TimeoutError:
        print(f"sympy timeout, model_answer: {model_answer}, ground_truth: {ground_truth}")
        return False
