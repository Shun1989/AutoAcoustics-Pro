import json
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path

import numpy as np

from autoacoustics.model import (AnalysisDetail, AnalysisResult, AnalysisSettings,
    AnalysisSummary, BatchItemSpec, CalibrationProfile, MeasurementContext,
    MeasurementProject, MetricResult, ResultSnapshot, ValidationError)
from autoacoustics.project import (file_hash, load_project, load_result_detail,
    relink_source, save_project, source_states)


class ProjectTests(unittest.TestCase):
    def session_fixture(self, folder):
        import h5py
        folder.mkdir(parents=True,exist_ok=True)
        data=folder/'samples.h5'
        with h5py.File(data,'w') as stream:
            samples=stream.create_dataset('samples',data=np.arange(16,dtype=float).reshape(1,16))
            samples.attrs.update({'sample_rate':48000,'channel_axis':0,'unit':'V'})
        manifest=folder/'session.json'
        payload={'schema_version':1,'session_id':'project-integrity-fixture','status':'complete','backend':'REPLAY',
            'is_simulated':True,'channels':[{'name':'mic1','unit':'V'}],'actual_sample_rate':48000,
            'requested_sample_rate':48000,'frames_expected':16,'frames_read':16,'frames_written':16,
            'flags':[],'errors':[],'events':[],'measurement_context':{},'calibration_applied_to_raw':False,
            'data_file':data.name,'data_sha256':file_hash(data),'sample_layout':'frames_by_channels_little_float64'}
        manifest.write_text(json.dumps(payload),encoding='utf-8')
        settings=AnalysisSettings(channel=0,start=0,end=16)
        first=BatchItemSpec(manifest,settings,input_hash=file_hash(manifest))
        second=replace(first,item_id='same-session-second-event',settings=replace(settings,start=4))
        result=AnalysisResult('session-result',first.input_hash,manifest,settings,first.context,None,AnalysisSummary())
        return MeasurementProject('DAQ项目',(first,second),(ResultSnapshot(result),)),data,payload

    def test_acquisition_nested_data_tamper_and_missing_invalidate_cached_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);project,data,payload=self.session_fixture(folder/'recording')
            self.assertEqual(set(source_states(project).values()),{'ok'})
            target=folder/'saved-project.json';save_project(project,target)
            data.write_bytes(b'changed nested recorded samples')
            self.assertEqual(set(source_states(project).values()),{'changed'})
            self.assertEqual(load_project(target).results[0].result.provenance['source_state'],'changed')
            data.unlink()
            self.assertEqual(set(source_states(project).values()),{'missing'})
            self.assertEqual(load_project(target).results[0].result.provenance['source_state'],'missing')

    def test_acquisition_relink_requires_matching_nested_data_and_updates_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);project,data,payload=self.session_fixture(folder/'original')
            moved=folder/'moved';project.items[0].path.parent.rename(moved);manifest=moved/'session.json'
            fixed=relink_source(project,project.items[0].item_id,manifest)
            self.assertEqual(source_states(fixed)[project.items[0].item_id],'ok')
            (moved/'samples.h5').write_bytes(b'tampered after moving')
            with self.assertRaises(ValidationError):relink_source(project,project.items[0].item_id,manifest)
            (moved/'samples.h5').unlink()
            with self.assertRaises(ValidationError):relink_source(project,project.items[0].item_id,manifest)

    def test_invalid_session_references_are_changed_and_ordinary_json_is_not_daq(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);project,data,payload=self.session_fixture(folder/'recording')
            manifest=project.items[0].path
            for reference in ('../outside.h5',str(data.resolve()),'nested/samples.h5'):
                payload['data_file']=reference;manifest.write_text(json.dumps(payload),encoding='utf-8')
                bad=replace(project,items=(replace(project.items[0],input_hash=file_hash(manifest)),))
                self.assertEqual(source_states(bad)[bad.items[0].item_id],'changed')
                with self.assertRaises(ValidationError):relink_source(bad,bad.items[0].item_id,manifest)
            ordinary=folder/'ordinary.json';ordinary.write_text(json.dumps({'schema_version':1,'data_file':'missing.h5','notes':'ordinary JSON'}),encoding='utf-8')
            item=replace(project.items[0],path=ordinary,input_hash=file_hash(ordinary));self.assertEqual(source_states(replace(project,items=(item,)))[item.item_id],'ok')

    def test_multiple_session_events_hash_each_manifest_and_data_only_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            project,data,payload=self.session_fixture(Path(temporary)/'recording')
            with patch('autoacoustics.project.file_hash',wraps=file_hash) as hashed:
                self.assertEqual(set(source_states(project).values()),{'ok'})
                targets=[Path(call.args[0]).resolve() for call in hashed.call_args_list]
                self.assertEqual(targets.count(project.items[0].path.resolve()),1)
                self.assertEqual(targets.count(data.resolve()),1)

    def fixture(self, folder):
        source = folder / '中文 录音.wav'
        source.write_bytes(b'unchanged original')
        profile = CalibrationProfile('reference', 'HEAD Pa reference', .8, 'FS')
        settings = AnalysisSettings(channel=0, start=1, end=10)
        item = BatchItemSpec(source, settings, profile, input_hash=file_hash(source))
        summary = AnalysisSummary({'LZeq': MetricResult(float('-inf'), 'dB', 'silent')})
        result = AnalysisResult('result-1', item.input_hash, source, settings,
            item.context, profile, summary,
            AnalysisDetail({'waveform': np.zeros(10)}, {'waveform': 'Pa'}))
        return MeasurementProject('座椅项目', (item,), (ResultSnapshot(result),))

    def test_project_reopens_summary_context_and_verified_lazy_detail(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            project = self.fixture(folder)
            target = folder / '测量.aap.json'
            save_project(project, target)
            reopened = load_project(target)
            self.assertEqual(reopened.items[0].item_id, project.items[0].item_id)
            self.assertEqual(reopened.items[0].settings.start, 1)
            self.assertEqual(reopened.items[0].context.actual_movement, 'unknown')
            self.assertEqual(reopened.results[0].result.summary.metrics['LZeq'].value, float('-inf'))
            self.assertIsNone(reopened.results[0].result.detail)
            detail = load_result_detail(reopened.results[0], target)
            np.testing.assert_array_equal(detail.arrays['waveform'], np.zeros(10))
            self.assertNotIn('waveform', target.read_text(encoding='utf-8'))
            self.assertEqual(source_states(reopened)[reopened.items[0].item_id], 'ok')

    def test_moved_and_changed_sources_are_not_silently_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            project = self.fixture(folder)
            old = project.items[0].path
            moved = folder / 'moved.wav'
            old.rename(moved)
            self.assertEqual(source_states(project)[project.items[0].item_id], 'missing')
            fixed = relink_source(project, project.items[0].item_id, moved)
            self.assertEqual(source_states(fixed)[fixed.items[0].item_id], 'ok')
            moved.write_bytes(b'different audio')
            self.assertEqual(source_states(fixed)[fixed.items[0].item_id], 'changed')
            with self.assertRaises(ValidationError):
                relink_source(project, project.items[0].item_id, moved)

    def test_corrupt_cache_and_unknown_schema_fail_without_guessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            target = folder / 'test.aap.json'
            save_project(self.fixture(folder), target)
            reopened = load_project(target)
            (folder / reopened.results[0].detail_cache).write_bytes(b'corrupt')
            with self.assertRaises(ValidationError):
                load_result_detail(reopened.results[0], target)
            data = json.loads(target.read_text(encoding='utf-8'))
            data['schema_version'] = 999
            target.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValidationError):
                load_project(target)


if __name__ == '__main__':
    unittest.main()
