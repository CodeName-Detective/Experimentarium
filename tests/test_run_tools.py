import argparse
import json
from pathlib import Path

import torch

from scripts.cleanup_runs import selected_records
from scripts.compare_runs import render_report
from scripts.evaluate_run import build_command
from scripts.export_checkpoint import export_state_dict
from scripts.plot_metrics import build_metric_series, render_html, write_csv
from src.main import _extract_config_file_args
from src.utils.run_inspect import config_path_for_run, run_status


def _write_registry(tmp_path: Path, run_id: str = 'run-a', mode: str = 'train') -> Path:
    run_dir = tmp_path / 'outputs' / 'runs' / run_id
    config_path = tmp_path / 'outputs' / 'run_configs' / f'{run_id}.yaml'
    metrics_path = run_dir / 'logs' / 'metrics.jsonl'
    checkpoint_dir = run_dir / 'checkpoints'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text('run:\n  name: test\n', encoding='utf-8')
    (checkpoint_dir / 'manifest.json').write_text('{"checkpoints": []}', encoding='utf-8')
    metrics_path.write_text(
        '\n'.join([
            json.dumps({'step': 1, 'metrics': {'train/loss': 1.0, 'val/accuracy': 0.5}}),
            json.dumps({'step': 2, 'metrics': {'train/loss': 0.5, 'val/accuracy': 0.75}}),
        ])
        + '\n',
        encoding='utf-8',
    )
    record = {
        'run_id': run_id,
        'config_id': 'abc123',
        'run_dir': 'outputs/runs/' + run_id,
        'config_path': 'outputs/run_configs/' + f'{run_id}.yaml',
        'command': 'uv run python src/main.py',
        'command_cwd': str(tmp_path),
        'config': {
            'run': {'mode': mode, 'runs_dir': 'outputs/runs'},
            'model': {'name': 'mlp'},
            'data': {'name': 'toy'},
            'task': {'name': 'classification'},
            'optimizer': {'name': 'adamw', 'lr': 0.001},
            'trainer': {'max_epochs': 1},
            'logging': {'jsonl': {'path': 'outputs/runs/' + run_id + '/logs/metrics.jsonl'}},
        },
    }
    registry = tmp_path / 'outputs' / 'run_registry.jsonl'
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(record) + '\n', encoding='utf-8')
    return registry


def test_config_path_for_run_and_main_from_run_flags(tmp_path, monkeypatch):
    registry = _write_registry(tmp_path)
    monkeypatch.chdir(tmp_path)
    expected_config = str(tmp_path / 'outputs' / 'run_configs' / 'run-a.yaml')

    assert config_path_for_run('run-a', registry) == Path(expected_config)
    config_file, overrides = _extract_config_file_args(['--from-run', 'run-a', '--run-id', 'copy'])
    assert config_file == expected_config
    assert overrides == ['run.id=copy']

    config_file, overrides = _extract_config_file_args(['--resume-run', 'run-a'])
    assert config_file == expected_config
    assert overrides == ['checkpoint.resume=latest', 'run.id=run-a']


def test_compare_runs_report_includes_final_and_best_metrics(tmp_path):
    registry = _write_registry(tmp_path)
    args = argparse.Namespace(
        registry=registry,
        run_ids=['run-a'],
        limit=0,
        metrics=['train/loss', 'val/accuracy'],
        config_fields=['model.name', 'optimizer.lr'],
        include_command=False,
        format='markdown',
        output=None,
    )

    report = render_report(args)

    assert 'run-a' in report
    assert 'final/train/loss' in report
    assert '0.5' in report
    assert '0.75' in report


def test_plot_metrics_writes_html_and_csv(tmp_path):
    registry = _write_registry(tmp_path)
    series = build_metric_series(['run-a'], registry, ['train/loss'])
    csv_path = tmp_path / 'metrics.csv'

    html = render_html(series)
    write_csv(series, csv_path)

    assert 'train/loss' in html
    assert 'run-a' in csv_path.read_text(encoding='utf-8')


def test_evaluate_run_builds_best_checkpoint_command(tmp_path):
    registry = _write_registry(tmp_path)
    args = argparse.Namespace(
        run_id='run-a',
        checkpoint='best',
        mode='eval',
        registry=registry,
        print_only=True,
        overrides=['trainer.limit_test_batches=1'],
    )

    command = build_command(args)

    assert '--config-file' in command
    assert str(tmp_path / 'outputs' / 'run_configs' / 'run-a.yaml') in command
    assert 'run.id=run-a' in command
    assert 'run.mode=eval' in command
    assert 'checkpoint.resume=best' in command
    assert command[-1] == 'trainer.limit_test_batches=1'


def test_cleanup_status_and_selection_detect_failed_runs(tmp_path):
    registry = _write_registry(tmp_path)
    run_dir = tmp_path / 'outputs' / 'runs' / 'run-a'
    (run_dir / 'checkpoints' / 'epoch_0001_exception.pt').write_text('failed', encoding='utf-8')
    records = selected_records(argparse.Namespace(registry=registry, run_ids=[], statuses=None, unsuccessful=True))

    assert len(records) == 1
    assert run_status(records[0]) == 'failed'


def test_export_state_dict_writes_model_only_payload(tmp_path):
    checkpoint = tmp_path / 'checkpoint.pt'
    output = tmp_path / 'model_state.pt'
    torch.save({'model_state': {'weight': torch.ones(1)}, 'epoch': 3, 'metrics': {'val/loss': 0.1}}, checkpoint)

    export_state_dict(checkpoint, output)

    payload = torch.load(output, map_location='cpu', weights_only=False)
    assert set(payload) >= {'model_state', 'epoch', 'metrics'}
    assert payload['epoch'] == 3
