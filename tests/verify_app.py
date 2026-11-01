"""Clicks through the app with Streamlit's test runner: every tab, every data source, every method.

    python tests/verify_app.py
"""

import os
import sys
import time

from streamlit.testing.v1 import AppTest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
sys.path.insert(0, HERE)


def btn(at, label):
    for b in at.button:
        if b.label == label:
            return b
    raise AssertionError(f"no button {label!r}; have {[b.label for b in at.button]}")


def check(at, where):
    if at.exception:
        raise AssertionError(f"{where}: {[e.value for e in at.exception]}")
    errs = [e.value for e in at.error]
    return errs


def texts(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.success] + [c.value for c in at.caption])


def main():
    t0 = time.time()
    at = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=600)
    at.run()
    check(at, "start")

    # --- example files: the upload path
    btn(at, "Try the example files").click().run()
    assert not check(at, "example"), at.error
    assert "Example files" in texts(at)
    btn(at, "Assign reviewers").click().run()
    assert not check(at, "assign example")
    assert "Proven optimal" in texts(at), "no certificate"
    rb = [m for m in at.metric if m.label == "Rules broken"][0]
    assert rb.value == "0", rb.value
    # rules: one per institution + an experienced reviewer per paper
    at.number_input(key="inst_cap").set_value(1).run()
    at.number_input(key="min_senior").set_value(1).run()
    check(at, "rules")
    assert any("run it again" in w.value for w in at.warning), "stale result not flagged"
    btn(at, "Compare all methods").click().run()
    check(at, "compare example")
    t = texts(at)
    assert "All methods on this data" in t
    print(f"  example files: ok ({time.time() - t0:.0f} s)")

    # --- re-plan after a drop-out
    ms = at.multiselect[0]
    ms.set_value([0]).run()
    btn(at, "Re-plan").click().run()
    check(at, "replan")
    ok = [s.value for s in at.success if s.value.startswith("Kept")]
    assert ok and "Rules broken: 0" in ok[0], ok
    print("  re-plan:", ok[0])

    # --- benchmark instance with bids
    at.selectbox[0].set_value("preflib1").run()
    btn(at, "Load").click().run()
    check(at, "preflib1")
    for m in ("fair", "randomized"):
        at.radio[0].set_value(m).run()
        btn(at, "Assign reviewers").click().run()
        check(at, f"preflib1 {m}")
        rb = [x for x in at.metric if x.label == "Rules broken"][0]
        assert rb.value == "0", (m, rb.value)
    print(f"  benchmark instance: ok ({time.time() - t0:.0f} s)")

    # --- demo with gold check, expertise search
    btn(at, "Load the demo").click().run()
    check(at, "demo")
    at.text_area[0].set_value("Graph neural networks for molecular property prediction with message passing").run()
    check(at, "abstract search")
    at.radio[0].set_value("max_total").run()
    btn(at, "Assign reviewers").click().run()
    check(at, "demo assign")
    t = texts(at)
    assert "Proven optimal" in t
    assert any(m.label == "Average self-rating, assigned pairs" for m in at.metric), "gold check missing"
    gm = [m for m in at.metric if m.label == "Average self-rating, assigned pairs"][0]
    print("  demo: ok, self-rating of assigned pairs", gm.value, gm.delta)
    # change expertise settings -> rebuild
    at.select_slider(key="temperature").set_value(float("inf")).run()
    check(at, "temperature")
    btn(at, "Assign reviewers").click().run()
    check(at, "demo assign 2")
    print(f"all app checks passed in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
