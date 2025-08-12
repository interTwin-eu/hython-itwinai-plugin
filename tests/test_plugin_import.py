def test_plugin_imports():
    from itwinai.plugins.hython.config import (
        HythonConfiguration,  # noqa: F401
    )
    from itwinai.plugins.hython.data import (
        RNNDatasetGetterAndPreprocessor,  # noqa: F401
        prepare_batch_for_device,  # noqa: F401
    )
    from itwinai.plugins.hython.trainer import (
        RNNDistributedTrainer,  # noqa: F401
    )
