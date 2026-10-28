import pytest

from studyscribe.system.power import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, keep_system_awake


class FakeSetState:
    def __init__(self, result: int = 0x80000000) -> None:
        self.calls: list[int] = []
        self.result = result

    def __call__(self, flags: int) -> int:
        self.calls.append(flags)
        return self.result


def test_requests_and_releases() -> None:
    fake = FakeSetState()
    with keep_system_awake(fake) as accepted:
        assert accepted
        assert fake.calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]
    assert fake.calls[-1] == ES_CONTINUOUS


def test_released_on_error() -> None:
    fake = FakeSetState()
    with pytest.raises(RuntimeError), keep_system_awake(fake):
        raise RuntimeError("capture failed")
    assert fake.calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]


def test_reports_refusal() -> None:
    fake = FakeSetState(result=0)
    with keep_system_awake(fake) as accepted:
        assert not accepted
    assert fake.calls[-1] == ES_CONTINUOUS


def test_real_call_on_this_machine() -> None:
    # On Windows this calls the real API; elsewhere it is a no-op.
    with keep_system_awake() as accepted:
        assert isinstance(accepted, bool)
