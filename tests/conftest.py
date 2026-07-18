"""Shared test fixtures.

Batch C — the parent-side inference gate (``mlx_soloheaven.inference_queue``)
is a process-wide singleton. Some endpoint tests drive the streaming handler
and deliberately DO NOT consume the returned ``StreamingResponse`` generator
(they only assert on the response type/status), which leaves the gate's
fast-path RUNNING slot held. An autouse reset gives every test a pristine gate
so that leak — or any gate state a queue test sets up — never contaminates a
later test (which would otherwise 429/503 or hang on a stale slot).

Finding 3 (codex): this reset previously MASKED a production bug — a pre-body
disconnect that leaked ``_running=1`` forever was silently wiped at teardown,
so no test ever noticed. The reset is now HYGIENE ONLY. A test that must
observe the real release on a pre-body disconnect marks itself
``@pytest.mark.no_gate_reset``; the autouse fixture then leaves the gate alone
so the test's own assertions are genuinely uncovered (a leak would fail the
assertion, not be swept under the rug). Such a test resets the gate itself at
the end for hygiene.
"""

import pytest

from mlx_soloheaven import inference_queue


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "no_gate_reset: do not auto-reset the inference gate (the test asserts "
        "real gate state — e.g. release on pre-body disconnect — without the "
        "reset masking it)",
    )


@pytest.fixture(autouse=True)
def _reset_inference_gate(request):
    # HYGIENE ONLY: a test that wants to observe real gate state (finding 3
    # regression) opts out so the reset cannot mask a leak.
    if request.node.get_closest_marker("no_gate_reset"):
        yield
        return
    inference_queue._reset_for_tests()
    yield
    inference_queue._reset_for_tests()
