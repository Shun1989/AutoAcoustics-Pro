from types import SimpleNamespace

import nidaqmx.constants as constants
from nidaqmx.errors import DaqError, DaqReadError
import numpy as np
import pytest

from autoacoustics.acquisition.device import AcquisitionConfig, AcquisitionError, AcquisitionStatus, ChannelConfig
from autoacoustics.acquisition.engine import RecordingEngine
from autoacoustics.acquisition.ni_daqmx import NiDaqmxBackend, discover_ni
from autoacoustics.acquisition.storage import open_acquisition_session, recover_session


class FakeChannels:
    def __init__(self):
        self.calls = []
    def add_ai_voltage_chan(self, **kwargs):
        self.calls.append(("voltage", kwargs))
        return SimpleNamespace(ai_min=kwargs["min_val"], ai_max=kwargs["max_val"],
                               ai_coupling=constants.Coupling.DC, ai_excit_val=0,
                               ai_excit_src=constants.ExcitationSource.NONE)
    def add_ai_microphone_chan(self, **kwargs):
        self.calls.append(("microphone", kwargs))
        return SimpleNamespace(ai_min=-1, ai_max=1, ai_coupling=constants.Coupling.AC,
                               ai_excit_val=kwargs["current_excit_val"], ai_excit_src=kwargs["current_excit_source"],
                               ai_microphone_sensitivity=kwargs["mic_sensitivity"],
                               ai_sound_pressure_db_ref=20e-6,
                               ai_sound_pressure_units=kwargs["units"],
                               ai_sound_pressure_max_sound_pressure_lvl=kwargs["max_snd_press_level"],
                               ai_rng_low=-5.0, ai_rng_high=5.0)


class FakeStream:
    def __init__(self, task):
        self.task = task
        self.total_samp_per_chan_acquired = 17
        self.overload_checks = []
    @property
    def overloaded_chans_exist(self):
        self.overload_checks.append("exists")
        if self.task.fault == "overload_unavailable":
            raise DaqError("fake unsupported overload detector", -200077)
        return self.task.fault == "analog_overload" or (self.task.fault == "stop_overload" and self.task.stopped)
    @property
    def overloaded_chans(self):
        assert self.overload_checks[-1] == "exists"
        self.overload_checks.append("channels")
        return ["Dev1/ai0"]
    @property
    def avail_samp_per_chan(self):
        return self.total_samp_per_chan_acquired-self.task.read_count


class FakeTask:
    def __init__(self, fault=None):
        self.fault, self.read_count = fault, 0
        self.ai_channels = FakeChannels()
        self.in_stream = FakeStream(self)
        self.timing = SimpleNamespace(samp_clk_rate=51200, cfg_samp_clk_timing=self.timing_config)
        self.started = self.stopped = self.closed = False
    def timing_config(self, **kwargs):
        if self.fault == "rate":
            raise DaqError("fake unsupported sampling rate", -200077)
        self.timing_args = kwargs
    def control(self, value):
        self.control_value = value
        if self.fault == "coerce_range":
            self.hardware.ai_max = 10
    def start(self):
        self.started = True
    def stop(self):
        if self.fault == "stop":
            raise DaqError("fake stop failure", -1)
        self.stopped = True
        if self.fault == "counter_reset":
            self.in_stream.total_samp_per_chan_acquired = 0
    def close(self):
        self.closed = True


class FakeReader:
    def __init__(self, stream):
        self.task = stream.task
    def read_many_sample(self, data, number_of_samples_per_channel, timeout):
        if self.task.fault == "overflow":
            raise DaqError("fake overflow", -200279)
        if self.task.fault == "disconnect":
            raise DaqError("fake disconnected", -201003)
        if self.task.fault == "timeout":
            raise DaqReadError("fake timeout", -200284, 0)
        if self.task.fault == "partial_timeout":
            self.task.fault = None
            count = 3
            data[:, :count] = np.arange(self.task.read_count, self.task.read_count+count)
            self.task.read_count += count
            raise DaqReadError("fake timeout with partial data", -200284, count)
        count = number_of_samples_per_channel
        for ch in range(data.shape[0]):
            data[ch, :count] = np.arange(self.task.read_count, self.task.read_count+count) + ch*100
        self.task.read_count += count
        return count


def fake_api(task=None, *, unavailable=False):
    task = task or FakeTask()
    device = SimpleNamespace(name="Dev1", product_type="FAKE MODEL", dev_serial_num=1234,
        ai_physical_chans=[SimpleNamespace(name="Dev1/ai0"), SimpleNamespace(name="Dev1/ai1")],
        ai_meas_types=[constants.UsageTypeAI.VOLTAGE, constants.UsageTypeAI.SOUND_PRESSURE_MICROPHONE],
        ai_voltage_rngs=[-5, 5], ai_min_rate=1000, ai_max_multi_chan_rate=51200, ai_max_single_chan_rate=51200)
    def local():
        if unavailable:
            raise RuntimeError("Could not find an installation of NI-DAQmx")
        return SimpleNamespace(devices=[device], driver_version=SimpleNamespace(major_version=99, minor_version=0, update_version=0))
    return SimpleNamespace(constants=constants, __version__="FAKE", Task=lambda: task,
                           system=SimpleNamespace(System=SimpleNamespace(local=local)))


def ni_config(**kwargs):
    return AcquisitionConfig((ChannelConfig("Dev1/ai0", "V1", "V", input_min=-5, input_max=5,
                                             coupling="AC", excitation_source="internal", excitation_current_a=0.0021),),
                             requested_sample_rate=48000, block_frames=8, target_frames=17, **kwargs)


def test_discovery_reports_missing_driver_without_simulated_fallback():
    result = discover_ni(api=fake_api(unavailable=True))
    assert not result.available and result.devices == () and "NI-DAQmx" in result.reason
    assert result.python_package_version == "FAKE"
    result = discover_ni(api=fake_api())
    assert result.available and result.devices[0].is_simulated
    assert result.devices[0].physical_channels == ("Dev1/ai0", "Dev1/ai1")


def test_ni_adapter_explicit_voltage_iepe_config_actual_rate_count_and_close(tmp_path):
    task = FakeTask()
    backend = NiDaqmxBackend(api=fake_api(task), reader_factory=FakeReader, is_simulated=True)
    engine = RecordingEngine(backend, ni_config(), tmp_path / "ni-fake")
    prepared = engine.configure()
    assert prepared.actual_sample_rate == 51200
    method, args = task.ai_channels.calls[0]
    assert method == "voltage" and args["min_val"] == -5 and args["max_val"] == 5
    assert task.timing_args["rate"] == 48000
    assert task.timing_args["sample_mode"] == constants.AcquisitionType.FINITE
    assert task.in_stream.overwrite == constants.OverwriteMode.DO_NOT_OVERWRITE_UNREAD_SAMPLES
    engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.COMPLETE and result.is_simulated
    assert result.frames_written == 17 and task.stopped and task.closed
    np.testing.assert_array_equal(open_acquisition_session(result.manifest_path).samples, [np.arange(17)])


def test_driver_scaled_pa_mic_requires_complete_explicit_parameters_and_stays_pa(tmp_path):
    task = FakeTask()
    mic = ChannelConfig("Dev1/ai0", "MicPa", "Pa", measurement_type="microphone",
        sensitivity_mv_pa=45, max_sound_pressure_db=110, excitation_source="internal", excitation_current_a=0.0021,
        sensor={"manufacturer": "PCB", "model": "user-confirmed"})
    conf = AcquisitionConfig((mic,), 48000, block_frames=8, target_frames=17)
    backend = NiDaqmxBackend(api=fake_api(task), reader_factory=FakeReader, is_simulated=True)
    prepared = backend.configure(conf)
    args = task.ai_channels.calls[0][1]
    assert args["mic_sensitivity"] == 45 and args["current_excit_val"] == 0.0021
    assert args["units"] == constants.SoundPressureUnits.PA
    assert prepared.metadata["channel_settings"][0]["driver_scaled"]
    actual = prepared.metadata["channel_settings"][0]
    assert actual["input_range_unit"] == "Pa"
    assert actual["microphone_sensitivity_mv_pa_actual"] == 45
    assert actual["sound_pressure_reference_pa_actual"] == 20e-6
    assert actual["sound_pressure_units_actual"] == "PA"
    assert actual["max_instantaneous_sound_pressure_db_actual"] == 110
    assert actual["adc_voltage_min_actual"] == -5.0 and actual["adc_voltage_max_actual"] == 5.0
    backend.close()
    invalid = ChannelConfig("Dev1/ai0", "MicPa", "Pa", measurement_type="microphone")
    with pytest.raises(AcquisitionError, match="灵敏度"):
        NiDaqmxBackend(api=fake_api(), reader_factory=FakeReader).configure(AcquisitionConfig((invalid,), 48000))


@pytest.mark.parametrize("attribute,value", [
    ("ai_microphone_sensitivity", 50), ("ai_microphone_sensitivity", float("nan")),
    ("ai_sound_pressure_db_ref", 1.0),
    ("ai_sound_pressure_units", constants.SoundPressureUnits.FROM_CUSTOM_SCALE),
    ("ai_sound_pressure_max_sound_pressure_lvl", float("inf")),
])
def test_unconfirmed_microphone_scaling_cannot_become_ready(attribute, value):
    task = FakeTask()
    original_add = task.ai_channels.add_ai_microphone_chan
    def add(**kwargs):
        hardware = original_add(**kwargs)
        setattr(hardware, attribute, value)
        return hardware
    task.ai_channels.add_ai_microphone_chan = add
    mic = ChannelConfig("Dev1/ai0", "MicPa", "Pa", measurement_type="microphone",
                        sensitivity_mv_pa=45, max_sound_pressure_db=110)
    with pytest.raises(AcquisitionError) as caught:
        NiDaqmxBackend(api=fake_api(task), reader_factory=FakeReader).configure(AcquisitionConfig((mic,), 48000))
    assert caught.value.flag == "microphone_scaling_unconfirmed"
    assert task.closed and not task.started


def test_microphone_readback_error_is_not_hidden_as_unknown():
    task = FakeTask()
    original_add = task.ai_channels.add_ai_microphone_chan
    class Hardware:
        @property
        def ai_microphone_sensitivity(self):
            raise DaqError("fake sensitivity readback error", -200077)
    def add(**kwargs):
        hardware = Hardware()
        hardware.__dict__.update(vars(original_add(**kwargs)))
        hardware.__dict__.pop("ai_microphone_sensitivity")
        return hardware
    task.ai_channels.add_ai_microphone_chan = add
    mic = ChannelConfig("Dev1/ai0", "MicPa", "Pa", measurement_type="microphone",
                        sensitivity_mv_pa=45, max_sound_pressure_db=110)
    with pytest.raises(AcquisitionError) as caught:
        NiDaqmxBackend(api=fake_api(task), reader_factory=FakeReader).configure(AcquisitionConfig((mic,), 48000))
    assert caught.value.code == -200077
    assert caught.value.flag == "microphone_scaling_unconfirmed" and task.closed


@pytest.mark.parametrize("fault,flag", [("overflow", "device_overflow"), ("disconnect", "device_disconnected"),
                                     ("timeout", "device_read_timeout"), ("partial_timeout", "device_read_timeout"),
                                     ("stop", "stop_failed"), ("counter_reset", "unknown_device_count")])
def test_native_errors_keep_codes_and_fail_measurement(tmp_path, fault, flag):
    task = FakeTask(fault)
    engine = RecordingEngine(NiDaqmxBackend(api=fake_api(task), reader_factory=FakeReader, is_simulated=True),
                             ni_config(), tmp_path / fault)
    engine.configure()
    engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.FAILED and flag in result.flags and result.errors
    assert task.closed
    if fault == "partial_timeout":
        assert result.frames_read == result.frames_written == 17  # timeout prefix was retained and drained
        assert any("-200284" in message for message in result.errors)


def test_unknown_physical_channel_and_raw_pa_relabelling_are_rejected():
    conf = AcquisitionConfig((ChannelConfig("Dev1/ai9", "bad", "V", input_min=-1, input_max=1),), 48000)
    with pytest.raises(AcquisitionError, match="通道"):
        NiDaqmxBackend(api=fake_api(), reader_factory=FakeReader).configure(conf)


def test_native_sampling_rate_rejection_keeps_error_code_and_closes_task():
    task = FakeTask("rate")
    with pytest.raises(AcquisitionError) as caught:
        NiDaqmxBackend(api=fake_api(task), reader_factory=FakeReader).configure(ni_config())
    assert caught.value.code == -200077 and task.closed


@pytest.mark.parametrize("fault,flag", [("analog_overload", "hardware_overload"),
                                      ("stop_overload", "hardware_overload"),
                                      ("overload_unavailable", "hardware_overload_detection_unavailable")])
def test_analog_overload_is_invalid_even_with_all_counts_and_low_digital_samples(tmp_path, fault, flag):
    task = FakeTask(fault)
    class LowLevelReader(FakeReader):
        def read_many_sample(self, data, **kwargs):
            count = super().read_many_sample(data, **kwargs)
            data[:, :count] *= 1e-4
            return count
    engine = RecordingEngine(NiDaqmxBackend(api=fake_api(task), reader_factory=LowLevelReader),
                             ni_config(), tmp_path / fault)
    engine.configure(); engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.FAILED
    assert result.frames_written == result.frames_read == result.frames_expected == 17
    assert flag in result.flags and result.errors and task.closed
    if fault != "overload_unavailable":
        assert any("Dev1/ai0" in error for error in result.errors)
        assert "channels" in task.in_stream.overload_checks
    else:
        assert any("-200077" in error for error in result.errors)
    with pytest.raises(AcquisitionError):
        open_acquisition_session(result.manifest_path)
    recovered = recover_session(result.manifest_path)
    assert recovered.status == AcquisitionStatus.FAILED and flag in recovered.flags
    assert np.max(np.abs(open_acquisition_session(result.manifest_path, allow_partial=True).samples)) < .002


def test_ni_stopped_hardware_counter_drains_buffer_without_restarting_task():
    task = FakeTask()
    conf = AcquisitionConfig((ChannelConfig("Dev1/ai0", "V", "V", input_min=-5, input_max=5),),
                             48000, block_frames=8)
    backend = NiDaqmxBackend(api=fake_api(task), reader_factory=FakeReader)
    backend.configure(conf)
    backend.start()
    first = backend.read_block(.25)
    assert first.first_sample_index == 0 and first.frames == 8
    assert backend.stop_acquisition() == 17
    tail = tuple(backend.drain_blocks())
    assert [block.first_sample_index for block in tail] == [8, 16]
    assert task.in_stream.auto_start is False and task.stopped and backend.read_count == 17
    backend.close()


def test_channel_actual_range_is_read_back_after_task_commit():
    task = FakeTask("coerce_range")
    original_add = task.ai_channels.add_ai_voltage_chan
    def add(**kwargs):
        task.hardware = original_add(**kwargs)
        return task.hardware
    task.ai_channels.add_ai_voltage_chan = add
    backend = NiDaqmxBackend(api=fake_api(task), reader_factory=FakeReader)
    prepared = backend.configure(ni_config())
    assert prepared.metadata["channel_settings"][0]["input_max_actual"] == 10
    backend.close()
    conf = AcquisitionConfig((ChannelConfig("Dev1/ai0", "bad", "Pa", input_min=-1, input_max=1),), 48000)
    with pytest.raises(AcquisitionError, match="V"):
        NiDaqmxBackend(api=fake_api(), reader_factory=FakeReader).configure(conf)


def _with_read_only_filter_delay(task, units, *, failing=None, invalid=None):
    """DAQmx injection is unavoidable here: this host has no NI driver/card."""
    original = task.ai_channels.add_ai_voltage_chan
    values = {"ai_filter_delay": 9876.5, "ai_filter_delay_units": units,
              "ai_remove_filter_delay": True, "ai_filter_delay_adjustment": -.25}
    if invalid is not None:
        values[invalid[0]] = invalid[1]
    reads, writes = [], []

    class Hardware:
        def __init__(self, base):
            object.__setattr__(self, "base", base)

        def __getattr__(self, name):
            if name in values:
                assert task.control_value == constants.TaskMode.TASK_COMMIT
                reads.append(name)
                if name == failing:
                    raise DaqError("filter delay getter unavailable", -200077)
                return values[name]
            return getattr(self.base, name)

        def __setattr__(self, name, value):
            if name in values:
                writes.append(name)
                raise AssertionError("filter delay must only be read")
            setattr(self.base, name, value)

    task.ai_channels.add_ai_voltage_chan = lambda **kwargs: Hardware(original(**kwargs))
    return reads, writes


@pytest.mark.parametrize("units", list(constants.DigitalWidthUnits))
def test_filter_delay_units_are_preserved_in_session_without_sample_or_time_correction(tmp_path, units):
    task = FakeTask()
    reads, writes = _with_read_only_filter_delay(task, units)
    engine = RecordingEngine(NiDaqmxBackend(api=fake_api(task), reader_factory=FakeReader),
                             ni_config(), tmp_path / units.name)
    prepared = engine.configure()
    readback = prepared.metadata["channel_settings"][0]["filter_delay_readback"]
    assert readback["applied_by_application"] is False
    assert readback["represents_full_measurement_chain"] is False
    assert readback["ai_filter_delay"]["raw_value"] == 9876.5
    assert readback["ai_filter_delay_adjustment"]["raw_value"] == -.25
    assert readback["ai_remove_filter_delay"]["raw_value"] is True
    assert readback["ai_filter_delay_units"]["raw_value"] == units.value
    assert readback["ai_filter_delay_units"]["enum_name"] == units.name
    for attribute in ("ai_filter_delay", "ai_filter_delay_units",
                      "ai_remove_filter_delay", "ai_filter_delay_adjustment"):
        assert readback[attribute]["status"] == "ok"
        assert readback[attribute]["error"] is None
        assert readback[attribute]["error_code"] is None
    engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.COMPLETE
    signal = open_acquisition_session(result.manifest_path)
    np.testing.assert_array_equal(signal.samples, [np.arange(17)])
    assert signal.sample_rate == 51200 and signal.time_origin == 0
    stored = signal.metadata["acquisition_session"]["configured"]["metadata"]["channel_settings"][0]
    assert stored["filter_delay_readback"] == readback
    assert len(reads) == 4 and not writes


@pytest.mark.parametrize("attribute", ["ai_filter_delay", "ai_filter_delay_units",
                                      "ai_remove_filter_delay", "ai_filter_delay_adjustment"])
def test_unavailable_filter_delay_getter_keeps_raw_recording_and_explicit_error(tmp_path, attribute):
    task = FakeTask()
    reads, writes = _with_read_only_filter_delay(task, constants.DigitalWidthUnits.TICKS, failing=attribute)
    engine = RecordingEngine(NiDaqmxBackend(api=fake_api(task), reader_factory=FakeReader),
                             ni_config(), tmp_path / attribute)
    engine.configure(); engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.COMPLETE and not result.errors
    signal = open_acquisition_session(result.manifest_path)
    np.testing.assert_array_equal(signal.samples, [np.arange(17)])
    assert signal.sample_rate == 51200 and signal.time_origin == 0
    readback = signal.metadata["acquisition_session"]["configured"]["metadata"]["channel_settings"][0]["filter_delay_readback"]
    assert readback[attribute]["status"] == "unavailable"
    assert readback[attribute]["raw_value"] is None
    assert readback[attribute]["error_code"] == -200077
    assert "filter delay getter unavailable" in readback[attribute]["error"]
    assert all(readback[name]["status"] == "ok" for name in reads if name != attribute)
    assert len(reads) == 4 and not writes


@pytest.mark.parametrize("attribute,value", [("ai_filter_delay", float("nan")),
                                            ("ai_filter_delay_adjustment", None)])
def test_invalid_filter_delay_value_is_unknown_not_zero_and_does_not_break_session(tmp_path, attribute, value):
    task = FakeTask()
    _with_read_only_filter_delay(task, constants.DigitalWidthUnits.SECONDS, invalid=(attribute, value))
    engine = RecordingEngine(NiDaqmxBackend(api=fake_api(task), reader_factory=FakeReader),
                             ni_config(), tmp_path / attribute)
    engine.configure(); engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.COMPLETE
    signal = open_acquisition_session(result.manifest_path)
    readback = signal.metadata["acquisition_session"]["configured"]["metadata"]["channel_settings"][0]["filter_delay_readback"]
    assert readback[attribute]["status"] == "unavailable"
    assert readback[attribute]["raw_value"] is None
    assert readback[attribute]["error"] and readback[attribute]["error_code"] is None
    np.testing.assert_array_equal(signal.samples, [np.arange(17)])


def test_confirmed_overload_survives_unavailable_channel_list_and_preserves_samples(tmp_path):
    task = FakeTask("analog_overload")
    class UnavailableChannelList(FakeStream):
        @property
        def overloaded_chans(self):
            assert self.overload_checks[-1] == "exists"
            raise DaqError("overloaded channel list unavailable", -200077)
    task.in_stream = UnavailableChannelList(task)
    engine = RecordingEngine(NiDaqmxBackend(api=fake_api(task), reader_factory=FakeReader),
                             ni_config(), tmp_path / "overload-known-list-unknown")
    engine.configure(); engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.FAILED
    assert "hardware_overload" in result.flags
    assert "hardware_overload_detection_unavailable" not in result.flags
    assert result.frames_read == result.frames_written == 17
    assert any("overloaded channel list unavailable" in error and "-200077" in error for error in result.errors)
    recover_session(result.manifest_path)
    np.testing.assert_array_equal(open_acquisition_session(result.manifest_path, allow_partial=True).samples,
                                  [np.arange(17)])
