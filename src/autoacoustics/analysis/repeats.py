"""Match saved measurement records without inferring independent trials."""
from dataclasses import dataclass, fields
from collections.abc import Mapping

from ..model import FrozenMap, to_plain


@dataclass(frozen=True)
class RepeatConditionMatch:
    matched: bool
    missing: tuple[str, ...]
    differences: tuple[str, ...]
    reference_conditions: Mapping


def _unknown(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {'', 'unknown', '未知', '未记录', '未校准'}
    return isinstance(value, (Mapping, tuple, list)) and not value


def _conditions(result):
    values = {f'context.{field.name}':getattr(result.context, field.name)
              for field in fields(result.context) if field.name not in {'repeat', 'notes'}}
    provenance = result.provenance
    values.update({
        'channel.index':result.settings.channel,
        'channel.name':provenance.get('channel_name'),
        'channel.source_unit':provenance.get('source_unit'),
        'channel.analysis_unit':provenance.get('unit'),
        'sampling.sample_rate':provenance.get('sample_rate'),
        'analysis.method_revision':provenance.get('method_revision'),
        'analysis.versions':to_plain(provenance.get('versions')),
    })
    # Sample positions belong to each recording. The event kind and processing
    # requests remain conditions; neither equal names nor equal duration prove
    # the same physical transient or an independent repetition.
    for field in fields(result.settings):
        if field.name not in {'start', 'end', 'channel'}:
            values['analysis.'+field.name] = getattr(result.settings, field.name)
    if result.calibration is not None:
        calibration = to_plain(result.calibration)
    else:
        calibration = to_plain(provenance.get('calibration'))
        if isinstance(calibration, dict):
            calibration.pop('ignored_profile', None)
    values['calibration.record'] = calibration
    missing = {key for key, value in values.items() if _unknown(value)}
    versions = values['analysis.versions']
    if isinstance(versions, Mapping) and any(_unknown(v) for v in versions.values()):
        missing.add('analysis.versions')
    if not isinstance(calibration, Mapping) or any(_unknown(calibration.get(key))
            for key in ('source', 'input_unit', 'coefficient')):
        missing.add('calibration.record')
    return values, missing


def compare_repeat_conditions(reference, candidate):
    """Compare only saved conditions; missing equal fields remain unverified.

    Calibration profiles use a conservative full snapshot comparison. Its
    identity/name/applicability differences are record differences, not proof
    that a physical coefficient changed. Pa declarations likewise do not prove
    a verified sensor chain or independent absolute calibration.
    """
    left, left_missing = _conditions(reference)
    right, right_missing = _conditions(candidate)
    differences = []
    for key, a in left.items():
        b = right[key]
        if _unknown(a) and _unknown(b):
            continue
        if a != b:
            differences.append(key)
    return RepeatConditionMatch(not differences, tuple(sorted(left_missing | right_missing)),
        tuple(differences), FrozenMap(left))
