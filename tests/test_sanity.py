from src.utils.sanity import run_sanity_checks


def test_sanity_checks_run(tiny_cfg):
    report = run_sanity_checks(tiny_cfg, strict=False)
    assert report.passed


def test_project_requirements_classify_packages_from_pyproject(monkeypatch, tmp_path):
    from src.utils.sanity import core

    (tmp_path / 'pyproject.toml').write_text(
        '''[project]
requires-python = ">=3.10"
dependencies = ["numpy>=1.24", "torch>=2.2"]

[project.optional-dependencies]
tracking = ["wandb>=0.16", "tensorboard>=2.15"]
vision = ["torchvision>=0.17"]
'''
    )
    monkeypatch.setattr(core, '_repo_root', lambda: tmp_path)

    requirements = core._load_project_requirements()

    assert requirements.requires_python == '>=3.10'
    assert requirements.dependencies == ('numpy>=1.24', 'torch>=2.2')
    assert set(requirements.optional_dependencies) == {'wandb>=0.16', 'tensorboard>=2.15', 'torchvision>=0.17'}


def test_project_requirements_do_not_inject_optional_fallbacks(monkeypatch, tmp_path):
    from src.utils.sanity import core

    (tmp_path / 'pyproject.toml').write_text(
        '''[project]
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]
'''
    )
    monkeypatch.setattr(core, '_repo_root', lambda: tmp_path)

    requirements = core._load_project_requirements()

    assert requirements.dependencies == ('numpy>=1.24',)
    assert requirements.optional_dependencies == ()


def test_missing_pyproject_skips_package_requirements(monkeypatch, tmp_path):
    from src.utils.sanity import core

    monkeypatch.setattr(core, '_repo_root', lambda: tmp_path)

    requirements = core._load_project_requirements()

    assert requirements.dependencies == ()
    assert requirements.optional_dependencies == ()
    assert 'package version checks skipped' in requirements.warning


def test_optional_dependency_missing_is_warning(monkeypatch, tiny_cfg):
    from src.utils.sanity import core

    monkeypatch.setattr(
        core,
        '_load_project_requirements',
        lambda: core.ProjectRequirements(dependencies=(), optional_dependencies=('definitely-missing-package>=1',)),
    )

    report = core.SanityReport()
    core._check_packages(report, tiny_cfg)

    [result] = report.results
    assert result.name == 'package.definitely-missing-package'
    assert not result.passed
    assert result.warning
    assert result.status == 'WARN'


def test_optional_dependency_import_success_has_no_failure_message(monkeypatch):
    from src.utils.sanity import core

    report = core.SanityReport()
    monkeypatch.setattr(core.importlib_metadata, 'version', lambda name: '0.20.0')
    monkeypatch.setattr(core, '_requirement_importable', lambda name: (True, name, ''))

    core._check_requirement(report, 'wandb>=0.16', warning=True)

    [result] = report.results
    assert result.name == 'package.wandb'
    assert result.passed
    assert not result.warning
    assert result.status == 'PASS'
    assert result.message == ''


def test_optional_dependency_import_failure_is_warning_not_pass(monkeypatch):
    from src.utils.sanity import core

    report = core.SanityReport()
    monkeypatch.setattr(core.importlib_metadata, 'version', lambda name: '0.20.0')
    monkeypatch.setattr(core, '_requirement_importable', lambda name: (False, name, 'broken import'))

    core._check_requirement(report, 'wandb>=0.16', warning=True)

    [result] = report.results
    assert result.name == 'package.wandb'
    assert not result.passed
    assert result.warning
    assert result.status == 'WARN'
    assert 'broken import' in result.message


def test_wandb_auto_check_is_skipped_when_logging_disabled(tiny_cfg):
    from src.utils.sanity import core

    report = core.SanityReport()
    core._check_wandb_runtime(report, tiny_cfg)

    assert not report.results


def test_wandb_online_logging_requires_api_key_and_network(monkeypatch, tiny_cfg):
    from src.utils.sanity import core

    cfg = dict(tiny_cfg)
    cfg['logging'] = {'wandb': {'enabled': True, 'mode': 'online'}}
    cfg['sanity'] = {'wandb': {'check': 'auto', 'timeout_seconds': 0.1}}
    monkeypatch.setattr(core, '_wandb_importable', lambda: (True, ''))
    monkeypatch.setattr(core, '_wandb_host', lambda: 'api.wandb.ai')
    monkeypatch.setattr(core, '_wandb_api_key_source', lambda host=None: None)
    monkeypatch.setattr(core, '_wandb_connectivity', lambda host, timeout: (False, 'network blocked'))

    report = core.SanityReport()
    core._check_wandb_runtime(report, cfg)

    results = {result.name: result for result in report.results}
    assert results['runtime.wandb.mode'].passed
    assert results['runtime.wandb.import'].passed
    assert not results['runtime.wandb.api_key'].passed
    assert not results['runtime.wandb.api_key'].warning
    assert not results['runtime.wandb.network'].passed
    assert not results['runtime.wandb.network'].warning


def test_wandb_offline_mode_does_not_require_key_or_network(monkeypatch, tiny_cfg):
    from src.utils.sanity import core

    cfg = dict(tiny_cfg)
    cfg['logging'] = {'wandb': {'enabled': True, 'mode': 'offline'}}
    cfg['sanity'] = {'wandb': {'check': 'auto'}}
    monkeypatch.setattr(core, '_wandb_importable', lambda: (True, ''))
    monkeypatch.setattr(core, '_wandb_api_key_source', lambda host=None: None)
    monkeypatch.setattr(core, '_wandb_connectivity', lambda host, timeout: (False, 'should not be called'))

    report = core.SanityReport()
    core._check_wandb_runtime(report, cfg)

    results = {result.name: result for result in report.results}
    assert results['runtime.wandb.mode'].passed
    assert results['runtime.wandb.import'].passed
    assert results['runtime.wandb.api_key'].passed
    assert results['runtime.wandb.network'].passed
    assert 'offline' in results['runtime.wandb.api_key'].message


def test_wandb_online_logging_passes_with_key_and_network(monkeypatch, tiny_cfg):
    from src.utils.sanity import core

    cfg = dict(tiny_cfg)
    cfg['logging'] = {'wandb': {'enabled': True, 'mode': 'online'}}
    cfg['sanity'] = {'wandb': {'check': 'auto'}}
    monkeypatch.setattr(core, '_wandb_importable', lambda: (True, ''))
    monkeypatch.setattr(core, '_wandb_host', lambda: 'api.wandb.ai')
    monkeypatch.setattr(core, '_wandb_api_key_source', lambda host=None: 'WANDB_API_KEY')
    monkeypatch.setattr(core, '_wandb_connectivity', lambda host, timeout: (True, 'reachable api.wandb.ai:443'))

    report = core.SanityReport()
    core._check_wandb_runtime(report, cfg)

    results = {result.name: result for result in report.results}
    assert results['runtime.wandb.api_key'].passed
    assert results['runtime.wandb.network'].passed


def test_minimum_driver_mapping_includes_cuda_13():
    from src.utils.sanity.cuda import minimum_nvidia_driver_for_cuda

    assert minimum_nvidia_driver_for_cuda('13.0') == '580.65.06'
    assert minimum_nvidia_driver_for_cuda('12.2') == '535.54.03'
    assert minimum_nvidia_driver_for_cuda(None) is None


def test_cuda_driver_mismatch_is_warning_in_cpu_mode(monkeypatch, tiny_cfg):
    from src.utils.sanity import core
    from src.utils.sanity.cuda import CudaDiagnostics

    diagnostics = CudaDiagnostics(
        torch_version='2.12.0+cu130',
        torch_cuda='13.0',
        cuda_available=False,
        cuda_warning='The NVIDIA driver on your system is too old (found version 12020).',
        device_count=0,
        device_names=(),
        nvidia_smi_available=True,
        nvidia_driver_version='535.183.01',
        nvidia_gpu_names=('NVIDIA Test GPU',),
        nvidia_smi_error='',
        min_driver_version='580.65.06',
        driver_compatible=False,
        compatibility_message='PyTorch CUDA build=13.0; NVIDIA driver=535.183.01; minimum driver for CUDA 13.0 is 580.65.06; torch.cuda.is_available()=False; driver is too old for this PyTorch CUDA build',
    )
    monkeypatch.setattr(core, 'collect_cuda_diagnostics', lambda: diagnostics)
    cfg = dict(tiny_cfg)
    cfg['run'] = dict(tiny_cfg['run'])
    cfg['run']['device'] = 'cpu'
    cfg['sanity'] = dict(tiny_cfg['sanity'])
    cfg['sanity']['cuda'] = {'check_driver': True, 'fail_on_cpu_mismatch': False}

    report = run_sanity_checks(cfg, strict=False)

    assert report.passed
    warnings = {result.name for result in report.warnings}
    assert 'runtime.cuda.driver_compatibility' in warnings
    assert 'runtime.cuda_available' in warnings


def test_model_smoke_reports_unavailable_cuda_device(monkeypatch, tiny_cfg):
    import torch

    from src.utils.sanity import core

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    cfg = dict(tiny_cfg)
    cfg['run'] = dict(tiny_cfg['run'])
    cfg['run']['device'] = 'cuda'

    report = core.SanityReport()
    core._check_data_model_smoke(report, cfg)

    result = {item.name: item for item in report.results}['smoke.device']
    assert not result.passed
    assert not result.warning
    assert 'cuda' in result.message


def test_cuda_driver_mismatch_fails_when_cuda_requested(monkeypatch, tiny_cfg):
    import torch

    from src.utils.sanity import core
    from src.utils.sanity.cuda import CudaDiagnostics

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    diagnostics = CudaDiagnostics(
        torch_version='2.12.0+cu130',
        torch_cuda='13.0',
        cuda_available=False,
        cuda_warning='The NVIDIA driver on your system is too old (found version 12020).',
        device_count=0,
        device_names=(),
        nvidia_smi_available=True,
        nvidia_driver_version='535.183.01',
        nvidia_gpu_names=('NVIDIA Test GPU',),
        nvidia_smi_error='',
        min_driver_version='580.65.06',
        driver_compatible=False,
        compatibility_message='PyTorch CUDA build=13.0; NVIDIA driver=535.183.01; minimum driver for CUDA 13.0 is 580.65.06; torch.cuda.is_available()=False; driver is too old for this PyTorch CUDA build',
    )
    monkeypatch.setattr(core, 'collect_cuda_diagnostics', lambda: diagnostics)
    cfg = dict(tiny_cfg)
    cfg['run'] = dict(tiny_cfg['run'])
    cfg['run']['device'] = 'cuda'
    cfg['sanity'] = dict(tiny_cfg['sanity'])
    cfg['sanity']['cuda'] = {'check_driver': True, 'fail_on_cpu_mismatch': False}

    report = run_sanity_checks(cfg, strict=False)

    failures = {result.name for result in report.failures}
    assert 'runtime.cuda.driver_compatibility' in failures
    assert 'runtime.cuda_available' in failures


def test_torch_install_recommendation_for_driver_535_prefers_cu121():
    from src.utils.sanity.cuda import CudaDiagnostics, format_torch_install_recommendation, recommend_torch_install

    diagnostics = CudaDiagnostics(
        torch_version='2.12.0+cu130',
        torch_cuda='13.0',
        cuda_available=False,
        cuda_warning='',
        device_count=0,
        device_names=(),
        nvidia_smi_available=True,
        nvidia_driver_version='535.309.01',
        nvidia_gpu_names=('NVIDIA Test GPU',),
        nvidia_smi_error='',
        min_driver_version='580.65.06',
        driver_compatible=False,
        compatibility_message='driver is too old',
    )

    recommendation = recommend_torch_install(diagnostics, python_requirement='>=3.10')
    message = format_torch_install_recommendation(recommendation)

    assert recommendation.python_for_uv == '3.10'
    assert recommendation.selected.cuda_tag == 'cu121'
    assert recommendation.selected.torch == '2.5.1'
    assert 'torch==2.5.1+cu121' in message
    assert 'https://download.pytorch.org/whl/cu121' in message
    assert 'python -m pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" --index-url https://download.pytorch.org/whl/cu121' in message


def test_cuda_diagnostics_recommendation_works_without_torch(monkeypatch):
    import subprocess

    from src.utils.sanity import cuda

    def fake_runner(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout='535.309.01, NVIDIA Test GPU\n',
            stderr='',
        )

    monkeypatch.setattr(cuda, '_import_torch', lambda: (None, "No module named 'torch'"))

    diagnostics = cuda.collect_cuda_diagnostics(runner=fake_runner)
    recommendation = cuda.recommend_torch_install(diagnostics, python_requirement='>=3.10')

    assert diagnostics.torch_version is None
    assert diagnostics.nvidia_driver_version == '535.309.01'
    assert diagnostics.nvidia_gpu_names == ('NVIDIA Test GPU',)
    assert 'PyTorch is not installed' in diagnostics.compatibility_message
    assert recommendation.selected.cuda_tag == 'cu121'
    assert 'torch==2.5.1+cu121' in cuda.format_torch_install_recommendation(recommendation)

