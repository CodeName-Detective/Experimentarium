import argparse
import json
from pathlib import Path

import torch

from scripts.cleanup_runs import selected_records
from scripts.compare_runs import render_report
from scripts.evaluate_run import build_command, parse_args
from scripts.export_checkpoint import export_state_dict
from scripts.plot_metrics import build_metric_series, render_html, write_csv
from src.main import _extract_config_file_args
from src.utils.run_inspect import config_path_for_run, run_status


def _write_registry(tmp_path: Path, run_id: str = 'run-a', mode: str = 'train') -> Path:
    trial_id = 2
    run_dir = tmp_path / 'outputs' / 'runs' / run_id / f'trial_{trial_id}'
    config_path = tmp_path / 'outputs' / 'run_configs' / run_id / f'trial_{trial_id}.yaml'
    metrics_path = run_dir / 'logs' / 'metrics.jsonl'
    checkpoint_dir = run_dir / 'checkpoints'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text('run:\n  name: test\n', encoding='utf-8')
    (checkpoint_dir / 'manifest.json').write_text('{"checkpoints": []}', encoding='utf-8')
    for name in ('last.pt', 'best.pt'):
        (checkpoint_dir / name).write_bytes(b'checkpoint')
    metrics_path.write_text(
        '\n'.join([
            json.dumps({'step': 1, 'metrics': {'train/loss': 1.0, 'val/accuracy': 0.5}}),
            json.dumps({'step': 2, 'metrics': {'train/loss': 0.5, 'val/accuracy': 0.75}}),
        ])
        + '\n',
        encoding='utf-8',
    )
    relative_run_dir = f'outputs/runs/{run_id}/trial_{trial_id}'
    record = {
        'run_id': run_id,
        'trial_id': trial_id,
        'config_id': 'abc123',
        'run_dir': relative_run_dir,
        'config_path': f'outputs/run_configs/{run_id}/trial_{trial_id}.yaml',
        'command': 'uv run python src/main.py',
        'command_cwd': str(tmp_path),
        'config': {
            'run': {
                'mode': mode,
                'runs_dir': 'outputs/runs',
                'run_dir': relative_run_dir,
                'trial_id': trial_id,
            },
            'model': {'name': 'mlp'},
            'data': {'name': 'toy'},
            'task': {'name': 'classification'},
            'optimizer': {'name': 'adamw', 'lr': 0.001},
            'trainer': {'max_epochs': 1},
            'logging': {'jsonl': {'path': f'{relative_run_dir}/logs/metrics.jsonl'}},
        },
    }
    registry = tmp_path / 'outputs' / 'run_registry.jsonl'
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(record) + '\n', encoding='utf-8')
    return registry


def test_config_path_for_run_and_main_from_run_flags(tmp_path, monkeypatch):
    registry = _write_registry(tmp_path)
    monkeypatch.chdir(tmp_path)
    expected_config = str(tmp_path / 'outputs' / 'run_configs' / 'run-a' / 'trial_2.yaml')
    expected_checkpoint = tmp_path / 'outputs' / 'runs' / 'run-a' / 'trial_2' / 'checkpoints' / 'last.pt'

    assert config_path_for_run('run-a', registry) == Path(expected_config)
    config_file, overrides = _extract_config_file_args(['--from-run', 'run-a', '--run-id', 'copy'])
    assert config_file == expected_config
    assert overrides == ['run.id=copy']

    config_file, overrides = _extract_config_file_args(['--resume-run', 'run-a'])
    assert config_file == expected_config
    assert overrides == [f'checkpoint.resume={expected_checkpoint}', 'run.mode=train']


def test_compare_runs_report_includes_trial_and_metrics(tmp_path):
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
    assert 'trial_id' in report
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


def test_evaluate_run_builds_explicit_best_checkpoint_command(tmp_path):
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
    expected_config = tmp_path / 'outputs' / 'run_configs' / 'run-a' / 'trial_2.yaml'
    expected_checkpoint = tmp_path / 'outputs' / 'runs' / 'run-a' / 'trial_2' / 'checkpoints' / 'best.pt'
    assert '--config-file' in command
    assert str(expected_config) in command
    assert not any(item.startswith('run.id=') for item in command)
    assert 'run.mode=eval' in command
    assert f'checkpoint.resume={expected_checkpoint}' in command
    assert command[-1] == 'trainer.limit_test_batches=1'


def test_evaluate_run_parser_accepts_intermixed_options_and_overrides():
    args = parse_args([
        'run-a',
        '--checkpoint',
        'best',
        'trainer.limit_test_batches=1',
        '--mode',
        'eval',
    ])
    assert args.run_id == 'run-a'
    assert args.checkpoint == 'best'
    assert args.mode == 'eval'
    assert args.overrides == ['trainer.limit_test_batches=1']


def test_cleanup_status_and_selection_detect_failed_runs(tmp_path):
    registry = _write_registry(tmp_path)
    run_dir = tmp_path / 'outputs' / 'runs' / 'run-a' / 'trial_2'
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


def test_resume_run_prefers_training_record_when_evaluation_is_newer(tmp_path, monkeypatch):
    registry = _write_registry(tmp_path)
    evaluation_config = tmp_path / 'outputs' / 'evaluations' / 'run-a' / 'trial_2' / 'eval_best' / 'config.yaml'
    evaluation_config.parent.mkdir(parents=True, exist_ok=True)
    evaluation_config.write_text('run:\n  mode: eval\n', encoding='utf-8')
    evaluation_record = {
        'run_id': 'run-a',
        'trial_id': 2,
        'config_path': str(evaluation_config),
        'run_dir': str(evaluation_config.parent),
        'command_cwd': str(tmp_path),
        'config': {'run': {'mode': 'eval', 'trial_id': 2}},
    }
    with registry.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(evaluation_record) + '\n')
    monkeypatch.chdir(tmp_path)

    config_file, overrides = _extract_config_file_args(['--resume-run', 'run-a'])
    expected_config = tmp_path / 'outputs' / 'run_configs' / 'run-a' / 'trial_2.yaml'
    expected_checkpoint = tmp_path / 'outputs' / 'runs' / 'run-a' / 'trial_2' / 'checkpoints' / 'last.pt'
    assert config_file == str(expected_config)
    assert overrides == [f'checkpoint.resume={expected_checkpoint}', 'run.mode=train']


def test_cleanup_unsuccessful_keeps_training_and_evaluation_records_separate(tmp_path):
    registry = _write_registry(tmp_path)
    training_dir = tmp_path / 'outputs' / 'runs' / 'run-a' / 'trial_2'
    (training_dir / 'checkpoints' / 'epoch_0001_exception.pt').write_text('failed', encoding='utf-8')
    evaluation_dir = tmp_path / 'outputs' / 'evaluations' / 'run-a' / 'trial_2' / 'eval_best'
    evaluation_metrics = evaluation_dir / 'logs' / 'metrics.jsonl'
    evaluation_metrics.parent.mkdir(parents=True, exist_ok=True)
    evaluation_metrics.write_text(json.dumps({'step': 1, 'metrics': {'test/loss': 0.1}}) + '\n', encoding='utf-8')
    evaluation_record = {
        'run_id': 'run-a',
        'trial_id': 2,
        'run_dir': 'outputs/evaluations/run-a/trial_2/eval_best',
        'command_cwd': str(tmp_path),
        'config': {
            'run': {'mode': 'eval', 'trial_id': 2},
            'logging': {'jsonl': {'path': 'outputs/evaluations/run-a/trial_2/eval_best/logs/metrics.jsonl'}},
        },
    }
    with registry.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(evaluation_record) + '\n')

    records = selected_records(argparse.Namespace(registry=registry, run_ids=[], statuses=None, unsuccessful=True))
    assert len(records) == 1
    assert run_status(records[0]) == 'failed'
    assert 'outputs/runs/run-a/trial_2' in str(records[0]['run_dir'])


def test_replay_and_resume_accept_custom_registry_from_any_position(tmp_path):
    registry = _write_registry(tmp_path)
    expected_config = str(tmp_path / 'outputs' / 'run_configs' / 'run-a' / 'trial_2.yaml')
    expected_checkpoint = tmp_path / 'outputs' / 'runs' / 'run-a' / 'trial_2' / 'checkpoints' / 'last.pt'

    config_file, overrides = _extract_config_file_args([
        '--resume-run',
        'run-a',
        '--registry',
        str(registry),
    ])
    assert config_file == expected_config
    assert overrides == [
        f'checkpoint.resume={expected_checkpoint}',
        'run.mode=train',
        f'run.config_registry={registry}',
    ]

    config_file, overrides = _extract_config_file_args([
        '--registry',
        str(registry),
        '--from-run',
        'run-a',
        '--run-id',
        'copy',
    ])
    assert config_file == expected_config
    assert overrides == ['run.id=copy', f'run.config_registry={registry}']
