from src.callbacks import LearningRateLogger, build_callbacks


def test_build_callbacks_from_config_list():
    callbacks = build_callbacks({'callbacks': [{'name': 'learning_rate_logger', 'every_n_steps': 2}]})

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], LearningRateLogger)


def test_build_callbacks_skips_disabled_entries():
    callbacks = build_callbacks({'callbacks': [{'name': 'learning_rate_logger', 'enabled': False}]})

    assert callbacks == []
