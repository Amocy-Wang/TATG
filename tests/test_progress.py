from io import StringIO

from tatg.progress import ProgressDisplay


class _PlainStream(StringIO):
    def isatty(self):
        return False


class _TTYStream(StringIO):
    def isatty(self):
        return True


def test_progress_recent_stats_logs_success_and_failing_test_counts():
    out = _PlainStream()
    progress = ProgressDisplay(
        total_targets=1,
        max_iterations=1,
        repair_iterations=0,
        worker_count=1,
        unit_label="method",
        java_version="11",
        junit_version="4.13.2",
        junit_family="junit4",
        enabled=True,
        out=out,
    )

    progress.register_targets(["Foo::bar"])
    progress.on_target_start("Foo::bar", "W1")
    progress.on_target_done("Foo::bar", True, 1.25, "W1")
    progress.on_target_test_stats("Foo::bar", 3, 2, 1)

    text = out.getvalue()
    assert "java=11" in text
    assert "junit=4.13.2 (junit4)" in text
    assert "generated=3" in text
    assert "tests_ok=2" in text
    assert "tests_fail=1" in text


def test_progress_repair_logs_candidate_progress_when_attempt_exceeds_configured_rounds():
    out = _PlainStream()
    progress = ProgressDisplay(
        total_targets=1,
        max_iterations=1,
        repair_iterations=1,
        worker_count=1,
        unit_label="method",
        enabled=True,
        out=out,
    )

    progress.register_targets(["Foo::bar"])
    progress.on_repair_start(1, 0, 4)
    progress.on_repair_end(1, 3, 1, 2)

    text = out.getvalue()
    assert "attempt=3 (candidates)" in text


def test_progress_coverage_logs_optional_scope_label():
    out = _PlainStream()
    progress = ProgressDisplay(
        total_targets=1,
        max_iterations=1,
        repair_iterations=1,
        worker_count=1,
        unit_label="method",
        enabled=True,
        out=out,
    )

    progress.on_coverage(0.545, 0.564, label="target_method")

    text = out.getvalue()
    assert "coverage  line=54.5%  branch=56.4%  scope=target_method" in text


def test_progress_stats_suppresses_empty_and_duplicate_non_tty_updates():
    out = _PlainStream()
    progress = ProgressDisplay(
        total_targets=1,
        max_iterations=1,
        repair_iterations=1,
        worker_count=1,
        unit_label="method",
        enabled=True,
        out=out,
    )

    progress.register_targets(["Foo::bar"])
    progress.on_target_start("Foo::bar", "W1")
    progress.on_target_done("Foo::bar", True, 0.5, "W1")
    progress.on_target_test_stats("Foo::bar", 0, 0, 0)
    progress.on_target_test_stats("Foo::bar", 2, 1, 1)
    progress.on_target_test_stats("Foo::bar", 2, 1, 1)

    text = out.getvalue()
    assert "generated=0" not in text
    assert text.count("generated=2") == 1


def test_progress_tty_validation_line_marks_awaiting_first_result():
    out = _TTYStream()
    progress = ProgressDisplay(
        total_targets=1,
        max_iterations=1,
        repair_iterations=0,
        worker_count=1,
        unit_label="method",
        enabled=True,
        out=out,
    )

    progress.register_targets(["Foo::bar"])
    progress.set_validation_workers(4)
    progress.on_validation_submitted(1)

    import re as _re
    # Strip ANSI escape sequences and cursor-move codes so we can check plain text.
    plain = _re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", _re.sub(r"\x1b\[[0-9;]*m", "", out.getvalue()))
    assert "[Generation + Validation]" in plain
    assert "0/1 jobs" in plain        # submitted=1, completed=0
    assert "1 in-flight" in plain
    assert "awaiting…" in plain       # no result received yet


def test_progress_tty_validation_line_clears_awaiting_marker_after_completion():
    out = _TTYStream()
    progress = ProgressDisplay(
        total_targets=1,
        max_iterations=1,
        repair_iterations=0,
        worker_count=1,
        unit_label="method",
        enabled=True,
        out=out,
    )

    progress.register_targets(["Foo::bar"])
    progress.set_validation_workers(2)
    progress.on_validation_submitted(1)
    progress.on_validation_completed(1)

    import re as _re
    # Strip ANSI and check only the final rendered frame (frames are separated by \r).
    plain = _re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", _re.sub(r"\x1b\[[0-9;]*m", "", out.getvalue()))
    frames = [f for f in plain.split("\r") if f.strip()]
    last = frames[-1] if frames else plain
    assert "1/1 jobs" in last         # submitted=1, completed=1
    assert "0 in-flight" in last
    assert "awaiting…" not in last    # cleared once a result arrived
