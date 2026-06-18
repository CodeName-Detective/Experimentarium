import pytest
import torch
from torch import nn

from src.engine.evaluator import Evaluator, move_to_device
from src.tasks import ClassificationTask, DetectionTask, LanguageModelingTask, RankingTask, SegmentationTask, build_task


class StaticOutputModel(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    def forward(self, batch):
        return self.outputs


@pytest.mark.parametrize(
    ('name', 'expected_type'),
    [
        ('segmentation', SegmentationTask),
        ('detection', DetectionTask),
        ('ranking', RankingTask),
        ('language_modeling', LanguageModelingTask),
    ],
)
def test_extended_tasks_are_registered(name, expected_type):
    task = build_task({'task': {'name': name, 'metrics': []}})
    assert isinstance(task, expected_type)


def test_segmentation_task_computes_pixel_loss_metrics_and_records():
    logits = torch.tensor(
        [[[[4.0, 1.0], [1.0, 4.0]], [[1.0, 4.0], [4.0, 1.0]]]],
        requires_grad=True,
    )
    batch = {'mask': torch.tensor([[[0, 1], [1, 255]]])}
    task = SegmentationTask({
        'output_key': 'logits',
        'target_key': 'mask',
        'ignore_index': 255,
        'loss': {'name': 'cross_entropy'},
        'metrics': ['accuracy'],
    })

    result = task.step(StaticOutputModel({'logits': logits}), batch, stage='train')

    assert result.loss is not None
    assert torch.isfinite(result.loss)
    assert task.compute_metrics()['accuracy'] == pytest.approx(1.0)
    records = task.predict_records(result.outputs, batch)
    assert records[0]['pred_mask'] == [[0, 1], [1, 0]]
    assert records[0]['target_mask'] == [[0, 1], [1, 255]]


def test_ranking_task_computes_loss_metrics_and_order():
    scores = torch.tensor([[0.2, 0.8, 0.4]], requires_grad=True)
    batch = {'relevance': torch.tensor([[0.0, 1.0, 0.5]])}
    task = RankingTask({
        'output_key': 'scores',
        'target_key': 'relevance',
        'loss': {'name': 'mse'},
        'metrics': ['mse', 'mae'],
    })

    result = task.step(StaticOutputModel({'scores': scores}), batch, stage='train')

    assert result.loss is not None
    assert torch.isfinite(result.loss)
    assert set(task.compute_metrics()) == {'mse', 'mae'}
    records = task.predict_records(result.outputs, batch)
    assert records[0]['ranking'] == [1, 2, 0]


def test_language_modeling_task_flattens_tokens_and_reports_perplexity():
    logits = torch.tensor(
        [[[4.0, 1.0, 0.0], [0.0, 4.0, 1.0], [1.0, 0.0, 4.0]]],
        requires_grad=True,
    )
    batch = {'labels': torch.tensor([[0, 1, -100]])}
    task = LanguageModelingTask({
        'output_key': 'logits',
        'target_key': 'labels',
        'ignore_index': -100,
        'loss': {'name': 'cross_entropy'},
        'metrics': ['accuracy'],
    })

    result = task.step(StaticOutputModel({'logits': logits}), batch, stage='train')

    assert result.loss is not None
    assert torch.isfinite(result.loss)
    metrics = task.compute_metrics()
    assert metrics['accuracy'] == pytest.approx(1.0)
    assert metrics['perplexity'] >= 1.0
    assert task.predict_records(result.outputs, batch)[0]['pred_tokens'] == [0, 1, 2]


def test_detection_task_aggregates_losses_and_filters_predictions():
    loss_classifier = torch.tensor(1.5, requires_grad=True)
    loss_box = torch.tensor(0.5, requires_grad=True)
    task = DetectionTask({
        'output_key': 'detections',
        'target_key': 'targets',
        'metrics': [],
        'score_threshold': 0.5,
        'nms_iou_threshold': 0.5,
    })
    batch = {'input': torch.randn(2, 3, 8, 8), 'targets': [{'boxes': torch.zeros(1, 4)}] * 2}
    result = task.step(
        StaticOutputModel({'losses': {'loss_classifier': loss_classifier, 'loss_box': loss_box}}),
        batch,
        stage='train',
    )

    assert result.loss is not None
    assert float(result.loss.detach()) == pytest.approx(2.0)
    assert result.targets is not None
    assert result.targets.shape == (2,)
    detections = {
        'detections': [
            {
                'boxes': torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.1, 0.1, 2.1, 2.1], [4.0, 4.0, 5.0, 5.0]]),
                'scores': torch.tensor([0.9, 0.8, 0.2]),
                'labels': torch.tensor([1, 1, 2]),
            }
        ]
    }
    records = task.predict_records(detections, batch)
    assert records == [{'boxes': [[0.0, 0.0, 2.0, 2.0]], 'scores': [0.8999999761581421], 'labels': [1]}]


def test_move_to_device_handles_nested_detection_batches():
    batch = {
        'input': [torch.ones(2), torch.zeros(2)],
        'targets': [{'boxes': torch.ones(1, 4), 'labels': torch.ones(1, dtype=torch.long)}],
    }

    moved = move_to_device(batch, 'cpu')

    assert moved['input'][0].device.type == 'cpu'
    assert moved['targets'][0]['boxes'].device.type == 'cpu'


def test_detection_evaluation_omits_loss_when_model_returns_only_predictions():
    task = DetectionTask({'output_key': 'detections', 'target_key': 'targets', 'metrics': []})
    model = StaticOutputModel({'detections': [{'boxes': torch.empty(0, 4), 'scores': torch.empty(0)}]})
    loader = [{'input': torch.randn(1, 3, 8, 8), 'targets': [{'boxes': torch.empty(0, 4)}]}]

    metrics = Evaluator(model, task).evaluate(loader, prefix='val')

    assert 'val/loss' not in metrics


def test_evaluator_uses_configured_precision_context_for_eval_and_predict(monkeypatch):
    calls = []

    class RecordingPrecisionContext:
        def __init__(self, device, precision):
            calls.append((str(device), precision))

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr('src.engine.evaluator.precision_autocast', RecordingPrecisionContext)
    task = DetectionTask({'output_key': 'detections', 'target_key': 'targets', 'metrics': []})
    model = StaticOutputModel({'detections': [{'boxes': torch.empty(0, 4), 'scores': torch.empty(0)}]})
    loader = [{'input': torch.randn(1, 3, 8, 8), 'targets': [{'boxes': torch.empty(0, 4)}]}]
    evaluator = Evaluator(model, task, precision='bf16')

    evaluator.evaluate(loader, prefix='val')
    evaluator.predict(loader)

    assert calls == [('cpu', 'bf16'), ('cpu', 'bf16')]


def test_detection_task_computes_map50_for_matching_boxes():
    task = DetectionTask({'output_key': 'detections', 'target_key': 'targets', 'metrics': ['map50']})
    batch = {
        'input': torch.randn(1, 3, 8, 8),
        'targets': [{'boxes': torch.tensor([[0.0, 0.0, 2.0, 2.0]]), 'labels': torch.tensor([1])}],
    }
    model = StaticOutputModel({
        'detections': [
            {
                'boxes': torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
                'scores': torch.tensor([0.9]),
                'labels': torch.tensor([1]),
            }
        ]
    })

    task.step(model, batch, stage='val')

    assert task.compute_metrics()['map50'] == pytest.approx(1.0)


def test_task_schema_validation_reports_missing_batch_key():
    task = ClassificationTask({
        'output_key': 'logits',
        'target_key': 'label',
        'loss': {'name': 'cross_entropy'},
        'metrics': [],
    })

    with pytest.raises(KeyError, match='missing required key'):
        task.step(StaticOutputModel({'logits': torch.randn(1, 2)}), {'input': torch.randn(1, 4)}, stage='train')
