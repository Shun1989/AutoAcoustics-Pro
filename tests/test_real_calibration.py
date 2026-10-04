import importlib.util
from pathlib import Path

def test_product_calibration_56_holdouts_with_one_saved_profile(tmp_path):
    path=Path(__file__).resolve().parents[1]/'tools/validate_real_calibration.py'
    specification=importlib.util.spec_from_file_location('real_calibration_validation',path)
    module=importlib.util.module_from_spec(specification);specification.loader.exec_module(module)
    result=module.validate(tmp_path/'validation.json')
    assert result['training_count']==8 and result['holdout_count']==56
    assert result['passed']
