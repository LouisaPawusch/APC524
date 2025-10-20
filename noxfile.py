import nox

@nox.session
def tests(session: nox.Session) -> None:
    """
    Run tests.
    """
    session.install(".[test]")
    session.run("pytest")
