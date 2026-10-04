from dataclasses import replace
from pathlib import Path
from autoacoustics.model import AnalysisResult,AnalysisSummary,AnalysisSettings,MetricResult,MeasurementContext

def result(value,unit='dB re 20 µPa',status='ok',**context):
    return AnalysisResult(str(value),'hash',Path('x.wav'),AnalysisSettings(channel=0),
        MeasurementContext(**context),None,AnalysisSummary({'LAeq':MetricResult(value,unit,status)}))

def test_right_minus_left_keeps_units_and_condition_differences():
    from autoacoustics.analysis.comparison import compare_results
    comparison=compare_results(result(50,supply='12 V'),result(54,supply='13 V'))
    assert comparison.differences['LAeq'].value==4
    assert comparison.differences['LAeq'].unit=='dB'
    assert any('supply' in x and '12 V' in x and '13 V' in x for x in comparison.condition_differences)
    assert any('unknown' in x for x in comparison.condition_differences)

def test_unavailable_incompatible_and_two_silences_never_become_zero_difference():
    from autoacoustics.analysis.comparison import compare_results
    assert compare_results(result(None,status='uncalibrated'),result(55)).differences['LAeq'].value is None
    assert compare_results(result(1,'V'),result(2,'Pa')).differences['LAeq'].value is None
    both=compare_results(result(float('-inf'),status='silent'),result(float('-inf'),status='silent'))
    assert both.differences['LAeq'].value is None
    one=compare_results(result(float('-inf'),status='silent'),result(50))
    assert one.differences['LAeq'].value==float('inf')
