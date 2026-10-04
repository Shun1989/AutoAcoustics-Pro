"""A record match must not invent physical trial or calibration evidence."""
import importlib
import importlib.util
from dataclasses import fields, replace
from pathlib import Path
import pytest

from autoacoustics.model import (AnalysisResult, AnalysisSettings, AnalysisSummary,
    CalibrationProfile, MeasurementContext, MetricResult)


def matcher():
    assert importlib.util.find_spec('autoacoustics.analysis.repeats') is not None, 'Condition matcher is missing.'
    return importlib.import_module('autoacoustics.analysis.repeats').compare_repeat_conditions


def result():
    context = MeasurementContext(**{f.name:f'known-{f.name}' for f in fields(MeasurementContext)
        if f.name not in {'repeat', 'notes'}})
    return AnalysisResult('first', 'source1', Path('first.wav'),
        AnalysisSettings(channel=0, start=10, end=1000, event_label='running'), context,
        None, AnalysisSummary({'RMS':MetricResult(.1, 'Pa')}), None,
        {'sample_rate':48000, 'channel_name':'Microphone', 'source_unit':'Pa', 'unit':'Pa',
         'method_revision':'revision-1', 'versions':{'numpy':'2.5', 'mosqito':'1.2.1'},
         'calibration':{'source':'文件声明 Pa', 'input_unit':'Pa', 'coefficient':1.,
                        'scaled_once':True, 'ignored_profile':None}})


def test_matching_records_allow_different_file_event_positions_and_repeat_notes():
    first = result()
    second = replace(first, result_id='second', input_hash='source2', path=Path('different.wav'),
        settings=replace(first.settings, start=100, end=2300),
        context=replace(first.context, repeat='2', notes='outlier note'))
    match = matcher()(first, second)
    assert match.matched
    assert not match.missing and not match.differences


@pytest.mark.parametrize('key,value', [
    ('sample_rate',51200), ('channel_name','Other microphone'), ('source_unit','V'),
    ('unit','FS'), ('method_revision','revision-2'), ('versions',{'numpy':'other'})])
def test_channel_units_sampling_and_algorithm_versions_exclude_different_records(key, value):
    first = result()
    second = replace(first, provenance={**first.provenance, key:value})
    match = matcher()(first, second)
    assert not match.matched and match.differences


def test_every_context_condition_except_repeat_and_notes_is_compared():
    first = result()
    for field in fields(first.context):
        if field.name in {'repeat', 'notes'}:
            continue
        second = replace(first, context=replace(first.context, **{field.name:'different'}))
        match = matcher()(first, second)
        assert not match.matched, field.name
        assert any(field.name in item for item in match.differences)


@pytest.mark.parametrize('change', [
    {'channel':1}, {'event_label':'startup'}, {'fft_size':8192}, {'window':'hamming'},
    {'field_type':'diffuse'}, {'remove_dc':True}, {'compute_loudness':False}])
def test_channel_event_and_processing_changes_do_not_match(change):
    first = result()
    second = replace(first, settings=replace(first.settings, **change))
    assert not matcher()(first, second).matched


def test_pa_passthrough_ignores_unapplied_profile_identity_but_checks_effective_source():
    first = result()
    second = replace(first, provenance={**first.provenance, 'calibration':{
        **first.provenance['calibration'], 'ignored_profile':'ignored-user-profile'}})
    assert matcher()(first, second).matched
    changed = replace(second, provenance={**second.provenance, 'calibration':{
        **second.provenance['calibration'], 'source':'different physical declaration'}})
    assert not matcher()(first, changed).matched


def test_unknown_values_match_only_as_incomplete_records_and_known_values_do_not_mix():
    first = replace(result(), context=MeasurementContext())
    second = replace(first, context=MeasurementContext(repeat='2', notes='same unknowns'))
    match = matcher()(first, second)
    assert match.matched and 'context.supply' in match.missing
    assert not matcher()(first, replace(second, context=replace(second.context, supply='12 V'))).matched
    for unknown in ('', 'unknown', '未知', None):
        absent = replace(first, provenance={**first.provenance, 'channel_name':unknown})
        assert 'channel.name' in matcher()(absent, absent).missing


def test_missing_calibration_and_dependency_metadata_are_not_confirmed():
    first = result()
    incomplete = replace(first, provenance={**first.provenance, 'versions':{}, 'calibration':{}})
    match = matcher()(incomplete, incomplete)
    assert match.matched
    assert 'analysis.versions' in match.missing and 'calibration.record' in match.missing


def test_calibration_profiles_and_summary_detail_use_effective_saved_records():
    first = result()
    profile = CalibrationProfile('Sensor', 'Sensor record', 2., 'V', mode='sensor_chain')
    first = replace(first, calibration=profile)
    assert matcher()(first, replace(first, provenance={**first.provenance,
        'methods':{'spectrum':{'fft_size':4096}}, 'include_detail':True})).matched
    assert not matcher()(first, replace(first, calibration=replace(profile, coefficient=3.))).matched
    assert not matcher()(first, replace(first, calibration=replace(profile, source='Other record'))).matched
