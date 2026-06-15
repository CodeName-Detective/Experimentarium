import pytest
import torch


@pytest.fixture
def tiny_cfg():
    return {
        'run': {'seed': 42, 'device': 'cpu', 'precision': 'fp32', 'deterministic': False, 'debug': False},
        'model': {'name': 'mlp', 'input_dim': 16, 'hidden_dim': 32, 'num_layers': 1, 'num_classes': 2, 'dropout': 0.0},
        'data': {
            'name': 'toy_classification',
            'seed': 42,
            'input_dim': 16,
            'num_classes': 2,
            'batch_size': 4,
            'num_workers': 0,
            'pin_memory': False,
            'splits': {'train': {'num_samples': 16}, 'val': {'num_samples': 8}, 'test': {'num_samples': 8}},
        },
        'task': {
            'name': 'classification',
            'output_key': 'logits',
            'target_key': 'label',
            'loss': {'name': 'cross_entropy'},
            'metrics': ['accuracy'],
        },
        'optimizer': {'name': 'adamw', 'lr': 1e-3, 'weight_decay': 0.0},
        'scheduler': {'name': 'none', 'interval': 'epoch', 'monitor': 'val/loss'},
        'trainer': {
            'max_epochs': 1,
            'accumulate_grad_batches': 1,
            'grad_clip': 1.0,
            'log_every_n_steps': 1,
            'val_every_n_epochs': 1,
            'early_stopping': {'patience': 0},
        },
        'checkpoint': {
            'dir': 'outputs/checkpoints/test',
            'save_every': 1,
            'keep_last_k': 2,
            'monitor': 'val/loss',
            'mode': 'min',
            'resume': None,
        },
        'logging': {'tensorboard': {'enabled': False}, 'wandb': {'enabled': False}},
        'sanity': {'strict': False, 'run_model_smoke': True, 'min_disk_gb': 0.0},
    }


@pytest.fixture
def tiny_batch():
    return {'input': torch.randn(4, 16), 'label': torch.zeros(4, dtype=torch.long)}
