import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_imports_when_cwd_is_not_project_root():
    os.chdir('/tmp')
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('app_under_test', ROOT / 'app.py')
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
        model, metadata = module.load_model_and_metadata()
    except FileNotFoundError as exc:
        raise AssertionError(f'La app falló al cargar archivos relativos: {exc}') from exc

    assert model is not None
    assert 'class_labels' in metadata


if __name__ == '__main__':
    test_app_imports_when_cwd_is_not_project_root()
    print('test_app_paths: OK')
