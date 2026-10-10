from __future__ import annotations

from scripts.analyze_humaneval import body_only, build_program, classify_failure, completion_of, has_whitespace_run, overlap, run_program

PROMPT = 'def add(a, b):\n    """Return a + b."""\n'
PROBLEM = {"task_id": "Test/0", "prompt": PROMPT, "entry_point": "add",
           "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n"}

def test_body_only_drops_code_after_the_function():
    generation = PROMPT.strip() + "\n    return a + b\n\nprint(add(\n"      
    assert body_only(generation, PROMPT) == PROMPT.strip() + "\n    return a + b\n"
    assert run_program(build_program(generation, PROBLEM)) == "failed: SyntaxError"
    assert run_program(build_program(body_only(generation, PROMPT), PROBLEM)) == "passed"

def test_run_program_statuses():
    assert run_program(build_program(PROMPT.strip() + "\n    return a - b\n", PROBLEM)) == "failed: AssertionError"
    assert run_program("while True:\n    pass\n", timeout_s=0.5) == "timeout"

def test_classify_failure_reads_the_last_traceback_line():
    assert classify_failure('Traceback (most recent call last):\n  File "p.py"\nIndentationError: unexpected indent\n') \
        == "IndentationError"
    assert classify_failure("Traceback\nmodule.CustomError: boom") == "CustomError"
    assert classify_failure("") == "unknown"

def test_whitespace_runs_are_counted_on_the_completion():
    assert has_whitespace_run(completion_of(PROMPT.strip() + "\n    return a + b" + "\n" * 60, PROMPT))
    assert not has_whitespace_run(completion_of(PROMPT.strip() + "\n    return a + b\n", PROMPT))

def test_overlap_counts():
    ref = [{"task_id": t, "status": s} for t, s in [("a", "passed"), ("b", "passed"), ("c", "failed: X"), ("d", "timeout")]]
    other = [{"task_id": t, "status": s} for t, s in [("a", "passed"), ("b", "failed: X"), ("c", "passed"), ("d", "timeout")]]
    o = overlap(ref, other)
    assert (o["both"], o["only_reference"], o["only_other"], o["neither"]) == (1, 1, 1, 1)
    assert o["only_reference_tasks"] == ["b"] and o["only_other_tasks"] == ["c"]