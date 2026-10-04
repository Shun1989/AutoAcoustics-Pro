"""Optional NI-DAQmx adapter with explicit channels and counter-based stop drain.

Real hardware/model acceptance is required: no driver installation or simulated
fallback occurs here. Injected APIs/readers are always labelled simulated.
"""
from __future__ import annotations

import importlib
import math
import time

import numpy as np

from .device import (AcquisitionError, ConfiguredAcquisition, DeviceCapabilities, DiscoveryResult, SampleBlock)


def _enum_name(value):
    return getattr(value, "name", str(value))


def _optional(obj, name):
    try:
        value = getattr(obj, name)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [_enum_name(item) if hasattr(item, "name") else item for item in value]
        return _enum_name(value)
    except Exception:
        return None


def _filter_delay_readback(hardware, constants):
    """Record committed AI filter state without changing acquisition timing."""
    records = {"applied_by_application": False,
               "represents_full_measurement_chain": False,
               "scope": "AI channel filter-delay state; not measured AO-to-AI, DUT or acoustic propagation delay"}
    for attribute in ("ai_filter_delay", "ai_filter_delay_units",
                      "ai_remove_filter_delay", "ai_filter_delay_adjustment"):
        record = {"status": "unavailable", "raw_value": None, "error": None, "error_code": None}
        try:
            value = getattr(hardware, attribute)
            if attribute == "ai_filter_delay_units":
                if not isinstance(value, constants.DigitalWidthUnits):
                    raise ValueError("DAQmx filter-delay unit is unknown")
                record.update(raw_value=value.value, enum_name=value.name)
            elif attribute == "ai_remove_filter_delay":
                if not isinstance(value, bool):
                    raise ValueError("DAQmx filter-delay removal state is unknown")
                record["raw_value"] = value
            else:
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError("DAQmx filter-delay value is unknown or not finite")
                record["raw_value"] = value
            record["status"] = "ok"
        except Exception as error:
            record.update(error=f"{type(error).__name__}: {error}", error_code=getattr(error, "error_code", None))
        records[attribute] = record
    return records


def _microphone_readback(hardware, requested, constants):
    """Confirm DAQmx's Pa scale after commit; these are measurement facts.

    ai_min/max use channel units (Pa here), whereas ai_rng_low/high describe
    the ADC voltage range. A missing/changed scale cannot be labelled Pa.
    """
    attributes = {"microphone_sensitivity_mv_pa_actual": "ai_microphone_sensitivity",
                  "sound_pressure_reference_pa_actual": "ai_sound_pressure_db_ref",
                  "max_instantaneous_sound_pressure_db_actual": "ai_sound_pressure_max_sound_pressure_lvl"}
    values = {}
    try:
        for key, attribute in attributes.items():
            value = float(getattr(hardware, attribute))
            if not math.isfinite(value):
                raise ValueError(f"{attribute} is not finite")
            values[key] = value
        units = hardware.ai_sound_pressure_units
        if units != constants.SoundPressureUnits.PA:
            raise ValueError("DAQmx sound pressure units are not Pa")
        if not math.isclose(values["microphone_sensitivity_mv_pa_actual"], requested.sensitivity_mv_pa,
                            rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("DAQmx microphone sensitivity differs from the requested mV/Pa")
        if not math.isclose(values["sound_pressure_reference_pa_actual"], 20e-6,
                            rel_tol=1e-10, abs_tol=1e-15):
            raise ValueError("DAQmx sound pressure reference differs from 20 micropascals")
        values["sound_pressure_units_actual"] = _enum_name(units)
        return values
    except Exception as error:
        raise AcquisitionError(f"NI 麦克风 Pa 缩放未确认：{error}；请核对灵敏度、单位和驱动设置。",
                               code=getattr(error, "error_code", None),
                               flag="microphone_scaling_unconfirmed") from error


def _api():
    api = importlib.import_module("nidaqmx")
    importlib.import_module("nidaqmx.system")
    importlib.import_module("nidaqmx.constants")
    return api


def discover_ni(*, api=None):
    injected = api is not None
    package_version = None
    try:
        api = api or _api()
        package_version = getattr(api, "__version__", None)
        system = api.system.System.local()
        version = system.driver_version
        driver = ".".join(str(getattr(version, key)) for key in ("major_version", "minor_version", "update_version"))
        devices = []
        for device in system.devices:
            channels = tuple(channel.name for channel in device.ai_physical_chans)
            metadata = {key: _optional(device, key) for key in ("ai_meas_types", "ai_voltage_rngs", "ai_min_rate",
                "ai_max_multi_chan_rate", "ai_max_single_chan_rate", "ai_couplings", "ai_current_int_excit_discrete_vals")}
            serial = _optional(device, "dev_serial_num")
            metadata.update({"serial_suffix": f"**{serial & 0xffff:04x}" if isinstance(serial, int) else None,
                             "capabilities_require_configuration_readback": True})
            devices.append(DeviceCapabilities(device.name, device.name, "NI_DAQMX", channels,
                driver, _optional(device, "product_type"), True, True, injected, metadata))
        return DiscoveryResult(True, tuple(devices), "" if devices else "NI-DAQmx 可用，但未发现模拟输入设备。",
                               driver, getattr(api, "__version__", None))
    except Exception as error:
        return DiscoveryResult(False, reason=f"NI-DAQmx 不可用：{error}。请核对 NI 驱动安装、设备连接和 NI MAX。",
                               python_package_version=package_version)


def _driver_error(error, default):
    code = getattr(error, "error_code", None)
    flag = {-200279: "device_overflow", -200361: "device_overflow", -200284: "device_read_timeout",
            -201003: "device_disconnected", -88705: "device_disconnected"}.get(code, default)
    return AcquisitionError(f"NI-DAQmx：{error}", code=code, flag=flag)


class NiDaqmxBackend:
    def __init__(self, *, api=None, reader_factory=None, is_simulated=False):
        self.api = api
        self.reader_factory = reader_factory
        self.is_simulated = bool(is_simulated or api is not None or reader_factory is not None)
        self.task = None
        self.read_count = 0
        self.started = self.closed = False
        self._stop_quality_error = None

    def _overload_error(self):
        try:
            # This getter clears the latch; the channel list is only valid
            # after this read. One acquisition reader owns both accesses.
            overloaded = self.task.in_stream.overloaded_chans_exist
            if not isinstance(overloaded, bool):
                raise ValueError("DAQmx overload detector did not return a boolean")
            if overloaded:
                try:
                    channels = self.task.in_stream.overloaded_chans
                except Exception as error:
                    return AcquisitionError(f"NI 已确认输入过载；过载通道列表读取失败：{error}；此段声压无效。",
                                            code=getattr(error, "error_code", None), flag="hardware_overload")
                return AcquisitionError(f"NI 输入过载：{', '.join(channels) or '通道未知'}；"
                                        "数字幅值和帧数完整不能证明此段声压有效。",
                                        flag="hardware_overload")
            return None
        except Exception as error:
            return AcquisitionError(f"NI 过载检测不可用：{error}；不能确认此测量的过载状态。",
                                    code=getattr(error, "error_code", None),
                                    flag="hardware_overload_detection_unavailable")

    def configure(self, config):
        discovery = discover_ni(api=self.api)
        if not discovery.available:
            raise AcquisitionError(discovery.reason, flag="driver_unavailable")
        selected = {channel.physical_channel.split("/")[0] for channel in config.channels}
        if len(selected) != 1:
            raise AcquisitionError("单会话须使用同一 NI 设备；跨设备同步需要另行验证。")
        devices = [d for d in discovery.devices if d.device_id in selected]
        if len(devices) != 1 or any(c.physical_channel not in devices[0].physical_channels for c in config.channels):
            raise AcquisitionError("所选 NI 设备或物理通道不可用。")
        self.api = self.api or _api()
        constants = self.api.constants
        try:
            self.task = self.api.Task()
            channel_settings = []
            hardware_channels = []
            for config_channel in config.channels:
                channel = config_channel
                if channel.measurement_type == "voltage":
                    if channel.unit != "V":
                        raise AcquisitionError("NI 电压通道只能记录源 V，不能重标为 Pa。")
                    if channel.input_min is None:
                        raise AcquisitionError("NI 电压通道须明确输入范围。")
                    hardware = self.task.ai_channels.add_ai_voltage_chan(physical_channel=channel.physical_channel,
                        name_to_assign_to_channel=channel.name, min_val=channel.input_min, max_val=channel.input_max,
                        units=constants.VoltageUnits.VOLTS)
                    hardware.ai_excit_voltage_or_current = constants.ExcitationVoltageOrCurrent.USE_CURRENT
                    hardware.ai_excit_src = getattr(constants.ExcitationSource, channel.excitation_source.upper())
                    if channel.excitation_current_a is not None:
                        hardware.ai_excit_val = channel.excitation_current_a
                elif channel.measurement_type == "microphone":
                    if channel.unit != "Pa" or channel.sensitivity_mv_pa is None or channel.max_sound_pressure_db is None:
                        raise AcquisitionError("NI 麦克风 Pa 通道须有明确灵敏度 mV/Pa、最大声压级和单位。")
                    hardware = self.task.ai_channels.add_ai_microphone_chan(physical_channel=channel.physical_channel,
                        name_to_assign_to_channel=channel.name, units=constants.SoundPressureUnits.PA,
                        mic_sensitivity=channel.sensitivity_mv_pa, max_snd_press_level=channel.max_sound_pressure_db,
                        current_excit_source=getattr(constants.ExcitationSource, channel.excitation_source.upper()),
                        current_excit_val=channel.excitation_current_a or 0.0)
                else:
                    raise AcquisitionError("NI 首版只配置明确的电压或麦克风通道。")
                if channel.coupling:
                    hardware.ai_coupling = getattr(constants.Coupling, channel.coupling)
                hardware_channels.append(hardware)
                channel_settings.append({"physical_channel": channel.physical_channel, "name": channel.name,
                    "unit": channel.unit, "measurement_type": channel.measurement_type,
                    "input_range_unit": channel.unit,
                    "driver_scaled": channel.measurement_type == "microphone", "calibration_applied_by_application": False})
            mode = constants.AcquisitionType.FINITE if config.target_frames is not None else constants.AcquisitionType.CONTINUOUS
            self.task.timing.cfg_samp_clk_timing(rate=config.requested_sample_rate, sample_mode=mode,
                samps_per_chan=config.target_frames or max(config.block_frames * config.queue_blocks * 2,
                                                          int(config.requested_sample_rate * 2)))
            self.task.in_stream.input_buf_size = max(config.block_frames * config.queue_blocks * 2,
                                                    int(config.requested_sample_rate * 2))
            self.task.in_stream.auto_start = False
            self.task.in_stream.relative_to = constants.ReadRelativeTo.CURRENT_READ_POSITION
            self.task.in_stream.offset = 0
            self.task.in_stream.overwrite = constants.OverwriteMode.DO_NOT_OVERWRITE_UNREAD_SAMPLES
            self.task.control(constants.TaskMode.TASK_COMMIT)
            actual = float(self.task.timing.samp_clk_rate)
            for setting, hardware, requested in zip(channel_settings, hardware_channels, config.channels):
                setting.update({name: _optional(hardware, attribute) for name, attribute in {
                    "input_min_actual": "ai_min", "input_max_actual": "ai_max", "coupling_actual": "ai_coupling",
                    "excitation_source_actual": "ai_excit_src", "excitation_current_a_actual": "ai_excit_val",
                    "terminal_configuration_actual": "ai_term_cfg"}.items()})
                setting.update({"adc_voltage_min_actual": _optional(hardware, "ai_rng_low"),
                                "adc_voltage_max_actual": _optional(hardware, "ai_rng_high"),
                                "adc_range_unit": "V"})
                setting["filter_delay_readback"] = _filter_delay_readback(hardware, constants)
                if requested.measurement_type == "microphone":
                    setting.update(_microphone_readback(hardware, requested, constants))
            self.config, self.rate = config, actual
            if self.reader_factory is None:
                from nidaqmx.stream_readers import AnalogMultiChannelReader
                self.reader_factory = AnalogMultiChannelReader
            self.reader = self.reader_factory(self.task.in_stream)
            device = devices[0]
            if self.is_simulated and not device.is_simulated:
                from dataclasses import replace
                device = replace(device, is_simulated=True)
            self.configured = ConfiguredAcquisition(device, config.requested_sample_rate, actual, config.channels,
                {"channel_settings": channel_settings, "clock": "DAQmx hardware sample clock",
                 "python_package_version": discovery.python_package_version,
                 "hardware_overload_detection": "DAQmx latch checked after every read and after stop; unknown invalidates recording",
                 "hardware_overload_detector_scope": "model-specific analog/digital coverage; sensor overload still requires chain headroom validation",
                 "main_storage": "application_chunk_journal_then_hdf5",
                 "native_tdms_throughput_comparison": "pending_real_hardware"})
            return self.configured
        except Exception as error:
            self.close()
            if isinstance(error, AcquisitionError):
                raise
            raise _driver_error(error, "configuration_failed") from error

    def start(self):
        if self.task is None or self.closed:
            raise AcquisitionError("NI 任务未配置。")
        try:
            self.task.start()
            self.started = True
        except Exception as error:
            raise _driver_error(error, "start_failed") from error

    def _read_count(self, count, timeout):
        values = np.empty((len(self.config.channels), count), dtype=np.float64)
        flags = ()
        error_code = error_message = None
        try:
            read = self.reader.read_many_sample(values, number_of_samples_per_channel=count, timeout=timeout)
        except Exception as error:
            partial = getattr(error, "samps_per_chan_read", 0)
            if getattr(error, "error_code", None) == -200284 and 0 < partial <= count:
                read, flags = partial, ("device_read_timeout",)
                error_code, error_message = error.error_code, str(error)
            else:
                raise _driver_error(error, "device_read_failed") from error
        if not isinstance(read, int) or not 0 < read <= count:
            raise AcquisitionError("NI 读取返回无效帧数。", flag="unknown_device_count")
        first = self.read_count
        self.read_count += read
        quality_error = self._overload_error()
        if quality_error is not None:
            flags = (*flags, quality_error.flag)
            message = str(quality_error)
            if quality_error.code is not None:
                message += f" [code={quality_error.code}]"
            error_message = f"{error_message}; {message}" if error_message else message
            if error_code is None:
                error_code = quality_error.code
        return SampleBlock(values[:, :read], first, first / self.rate, flags, error_code, error_message)

    def read_block(self, timeout):
        if self.config.target_frames is not None and self.read_count >= self.config.target_frames:
            return None
        deadline = time.monotonic() + timeout
        try:
            while self.task.in_stream.avail_samp_per_chan <= 0:
                if time.monotonic() >= deadline:
                    raise AcquisitionError("NI 在读取期限内未提供样本。", flag="device_read_timeout")
                time.sleep(min(0.002, timeout))
            count = min(self.config.block_frames, self.task.in_stream.avail_samp_per_chan)
            if self.config.target_frames is not None:
                count = min(count, self.config.target_frames-self.read_count)
            return self._read_count(count, timeout)
        except AcquisitionError:
            raise
        except Exception as error:
            raise _driver_error(error, "device_read_failed") from error

    def stop_acquisition(self):
        try:
            self.task.stop()
            self.stop_count = int(self.task.in_stream.total_samp_per_chan_acquired)
            self._stop_quality_error = self._overload_error()
            if self.stop_count < self.read_count:
                raise AcquisitionError("NI 停止后的累计采样计数小于已读数，不能确认完整性。", flag="unknown_device_count")
            return self.stop_count
        except AcquisitionError:
            raise
        except Exception as error:
            raise _driver_error(error, "stop_failed") from error

    def drain_blocks(self):
        while self.read_count < self.stop_count:
            count = min(self.config.block_frames, self.stop_count-self.read_count)
            yield self._read_count(count, self.config.read_timeout)
        # A late overload may occur after the final normal block and leave
        # no unread samples. Drain first, then invalidate the measurement.
        if self._stop_quality_error is not None:
            raise self._stop_quality_error

    def close(self):
        if self.task is not None and not self.closed:
            try:
                self.task.close()
            finally:
                self.closed = True
