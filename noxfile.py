from __future__ import annotations

import nox


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run tests.
    """
    session.install(".[test]")
    session.run("pytest")
