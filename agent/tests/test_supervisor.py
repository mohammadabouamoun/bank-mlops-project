from agent.app.services.supervisor import supervisor_flow


def test_high_severity():
    result = supervisor_flow("high")

    assert result["action"] == "retrain_model"
    assert result["requires_approval"] is True


def test_medium_severity():
    result = supervisor_flow("medium")

    assert result["action"] == "log_only"
    assert result["requires_approval"] is False


def test_low_severity():
    result = supervisor_flow("low")

    assert result["action"] == "do_nothing"
    assert result["requires_approval"] is False