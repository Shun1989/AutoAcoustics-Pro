"""Generic MAT/HDF5/TDMS signals; no vendor-specific layout guesses."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
from nptdms import TdmsFile
from scipy.io import loadmat
from scipy.io.matlab import MatReadError

from autoacoustics.model import ChannelInfo, ImportError, ImportMapping, MappingRequired
from .common import make_signal, plain_metadata, positive_rate, resolve_units


METADATA_KEYS = {"fs", "sample_rate", "sampling_rate", "unit", "units", "physical_unit", "channel_names",
                 "channel_axis", "wf_increment", "scale_factor", "add_offset", "scale_applied",
                 "scaling_convention", "time_origin_seconds"}


def _scalar(value):
    if value is None:
        return None
    array = np.asarray(value)
    return array.reshape(-1)[0].item() if array.size == 1 else value


def _sample_rate(attributes, mapping, options):
    if mapping and mapping.sample_rate is not None:
        return positive_rate(mapping.sample_rate)
    candidates = [_scalar(attributes[key]) for key in ("sample_rate", "sampling_rate", "fs") if key in attributes]
    if "wf_increment" in attributes:
        candidates.append(1.0 / positive_rate(_scalar(attributes["wf_increment"])))
    if not candidates:
        raise MappingRequired("科学文件缺少明确采样率，请在映射中填写。", {**options, "needs_sample_rate": True})
    rates = [positive_rate(value) for value in candidates]
    if not np.allclose(rates, rates[0], rtol=1e-10):
        raise MappingRequired("科学文件采样率元数据互相冲突，请明确填写采样率。", {**options, "needs_sample_rate": True})
    return rates[0]


def _array_signal(path, value, dataset, attributes, mapping, *, format_name, extra_metadata=None):
    value = np.asarray(value)
    options = {"datasets": [dataset], "shape": list(value.shape)}
    if value.dtype.kind not in "iuf" or value.ndim > 2:
        raise ImportError("所选数据集须为一维或二维的实数数值数组；不忽略复数的虚部。")
    source_dtype = str(value.dtype)
    if value.ndim == 0:
        samples = value.reshape(1, 1)
        axis = 0
    elif value.ndim == 1:
        samples = value.reshape(1, -1)
        axis = None
    else:
        axis = mapping.channel_axis if mapping and mapping.channel_axis is not None else _scalar(attributes.get("channel_axis"))
        if axis is None and 1 in value.shape:
            # A stored row/column vector may be a single channel; an explicit
            # axis always takes precedence, including multi-channel one-frame.
            samples = value.reshape(1, -1)
        elif axis in (0, 1):
            samples = value if axis == 0 else value.T
        else:
            raise MappingRequired("二维数组须明确哪个轴为通道；不按数组尺寸猜测。",
                                  {**options, "needs_channel_axis": True})
    count = samples.shape[0]
    rate = _sample_rate(attributes, mapping, options)
    declared_units = attributes.get("units", attributes.get("unit", attributes.get("physical_unit")))
    units = resolve_units(count, mapping, declared_units, options=options)
    names = (list(mapping.channels) if mapping and mapping.channels
             else plain_metadata(attributes.get("channel_names", [f"Channel {index + 1}" for index in range(count)])))
    if isinstance(names, str):
        names = [names]
    names = [str(name).strip() for name in names]
    if len(names) != count or any(not name for name in names):
        raise ImportError("通道名称数量与所选矩阵不一致。")
    scaling = {"applied_by_importer": False, "source_scale_applied": plain_metadata(attributes.get("scale_applied"))}
    if "scale_factor" in attributes or "add_offset" in attributes:
        # A plain arbitrary HDF attribute is not an instrument contract. Only
        # accept an explicitly declared convention/application state.
        already = _scalar(attributes.get("scale_applied"))
        convention = str(plain_metadata(_scalar(attributes.get("scaling_convention", ""))))
        if already is None or convention not in {"linear", "CF"}:
            raise MappingRequired("缩放字段的含义或是否已应用未声明；请补充 scale_applied 与 scaling_convention。", options)
        if not isinstance(already, (bool, int, np.integer)) or already not in (0, 1):
            raise ImportError("科学文件缩放状态 scale_applied 须明确为布尔值或 0/1。")
        slope = float(_scalar(attributes.get("scale_factor", 1.0)))
        offset = float(_scalar(attributes.get("add_offset", 0.0)))
        if not np.isfinite(slope) or not np.isfinite(offset):
            raise ImportError("科学文件缩放系数不是有限数值。")
        scaling.update({"convention": convention, "scale_factor": slope, "add_offset": offset})
        if not bool(already):
            samples = samples.astype(float) * slope + offset
            scaling["applied_by_importer"] = True
    channels = tuple(ChannelInfo(name, unit, "sound_pressure" if unit == "Pa" else "unknown",
                                 scaled=unit in {"V", "Pa", "m/s²", "g"},
                                 metadata={"source_dataset": dataset, "source_unit": plain_metadata(declared_units)})
                     for name, unit in zip(names, units))
    quality = ["unknown_source_unit"] if "unknown" in units else []
    if "acquisition_complete" in attributes:
        completion = _scalar(attributes["acquisition_complete"])
        if not isinstance(completion, (bool, int, np.integer)) or completion not in (0, 1):
            raise ImportError("采集数据的完整状态 acquisition_complete 须明确为布尔值或 0/1。")
        if not completion:
            quality.append("incomplete_recording")
    return make_signal(path, samples, rate, channels, mapping=mapping,
                       quality=quality,
                       metadata={"format": format_name, "dataset": dataset, "source_dtype": source_dtype,
                                 "logical_source_shape": list(value.shape), "channel_axis": axis,
                                 "source_attributes": plain_metadata(dict(attributes)), "scaling": scaling,
                                 "time_origin_seconds": plain_metadata(attributes.get("time_origin_seconds", 0.0)),
                                 **(extra_metadata or {})})


def _select_dataset(candidates, mapping):
    names = list(candidates)
    selected = mapping.dataset if mapping else None
    if selected is not None:
        if selected not in candidates:
            raise ImportError(f"所选变量或数据集不存在：{selected}")
        return selected
    if len(names) != 1:
        raise MappingRequired("科学文件须明确选择一个数值变量或数据集。", {"datasets": names})
    return names[0]


def import_mat(path, mapping):
    try:
        # Keep matrix dimensions: squeeze_cells would turn two channels with
        # one frame into a one-channel vector and discard the declared axis.
        contents = loadmat(path, struct_as_record=False, squeeze_me=False)
    except (ValueError, TypeError, OSError, NotImplementedError, MatReadError) as error:
        raise ImportError(f"无法读取 MAT 文件：{error}") from error
    candidates, parents = {}, {}

    def struct_fields(value):
        if hasattr(value, "_fieldnames"):
            return {name: getattr(value, name) for name in value._fieldnames}
        if isinstance(value, np.ndarray) and value.dtype.kind == "O" and value.size == 1:
            return struct_fields(value.reshape(-1)[0])
        return value

    def walk(values, prefix="", inherited=None):
        attributes = dict(inherited or {})
        attributes.update({key: item for key, item in values.items() if key in METADATA_KEYS})
        for name, value in values.items():
            if name.startswith("__") or name in METADATA_KEYS:
                continue
            key = prefix + name
            value = struct_fields(value)
            if isinstance(value, Mapping):
                walk(value, key + ".", attributes)
            elif isinstance(value, (np.ndarray, list, tuple, int, float, np.number)):
                array = np.asarray(value)
                if array.dtype.kind in "iufc" and array.ndim <= 2:
                    candidates[key], parents[key] = array, attributes
    walk(contents)
    key = _select_dataset(candidates, mapping)
    return _array_signal(path, candidates[key], key, parents[key], mapping, format_name="MAT")


def _hdf_value(dataset):
    value = dataset[()]
    matlab_class = _scalar(dataset.attrs.get("MATLAB_class"))
    if matlab_class in {"char", b"char"} and np.asarray(value).dtype.kind in "iu":
        return "".join(chr(int(code)) for code in np.asarray(value).reshape(-1, order="F") if code)
    return plain_metadata(value)


def import_hdf5(path, mapping):
    try:
        with h5py.File(path, "r") as contents:
            candidates = {}
            def visit(name, value):
                if isinstance(value, h5py.Dataset) and value.dtype.kind in "iufc" and value.ndim <= 2:
                    if Path(name).name not in METADATA_KEYS and _scalar(value.attrs.get("MATLAB_class")) not in {"char", b"char"}:
                        candidates["/" + name] = value
            contents.visititems(visit)
            # Accept either canonical /nested/path or an explicit path without /.
            if mapping and mapping.dataset and not mapping.dataset.startswith("/"):
                names = {name.lstrip("/"): value for name, value in candidates.items()}
                chosen = _select_dataset(names, mapping)
                key = "/" + chosen
            else:
                key = _select_dataset(candidates, mapping)
            dataset = candidates[key]
            ancestors = [contents]
            group = dataset.parent
            while group.name != "/":
                ancestors.append(group)
                group = group.parent
            attributes = {}
            for node in [contents, *reversed(ancestors[1:]), dataset]:
                attributes.update(dict(node.attrs))
                if isinstance(node, h5py.Group):
                    for name in METADATA_KEYS:
                        if name in node and isinstance(node[name], h5py.Dataset):
                            attributes[name] = _hdf_value(node[name])
            value = dataset[()]
            matlab_class = _scalar(dataset.attrs.get("MATLAB_class"))
            matlab = path.suffix.lower() == ".mat" and matlab_class is not None
            storage_transpose = matlab and value.ndim == 2
            if storage_transpose:
                value = value.T
            return _array_signal(path, value, key, attributes, mapping,
                                 format_name="MAT_7.3" if matlab else "HDF5",
                                 extra_metadata={"matlab_storage_transpose": storage_transpose,
                                                 "stored_source_shape": list(dataset.shape)})
    except ImportError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ImportError(f"无法读取通用 HDF5 数值布局：{error}") from error


def import_tdms(path, mapping):
    try:
        contents = TdmsFile.read(path)
        if contents.file_status.incomplete_final_segment:
            raise ImportError("TDMS 最后一个数据段不完整，不能当作完整测量导入。")
        candidates = {channel.path: channel for group in contents.groups() for channel in group.channels()
                      if len(channel) and channel.dtype.kind in "iufc"}
        options = {"channels": list(candidates)}
        if not candidates:
            raise ImportError("TDMS 没有可读取的数值通道。")
        selected = list(mapping.channels) if mapping and mapping.channels else list(candidates)
        if len(set(selected)) != len(selected) or any(name not in candidates for name in selected):
            raise ImportError("所选 TDMS 完整通道路径不存在或重复。")
        if mapping and mapping.dataset:
            selected = [name for name in selected if candidates[name].group_name == mapping.dataset]
            if not selected:
                raise ImportError("所选 TDMS 分组没有数值通道。")
        sources = [candidates[name] for name in selected]
        arrays = [np.asarray(channel[:]) for channel in sources]
        if any(array.dtype.kind not in "iuf" for array in arrays):
            raise ImportError("TDMS 通道必须是实数信号。")
        if len({array.size for array in arrays}) != 1:
            raise MappingRequired("TDMS 通道帧数不一致，请选择共用时间轴的通道。", options)
        rates = [_sample_rate(channel.properties, mapping, options) for channel in sources]
        origins = [(plain_metadata(channel.properties.get("wf_start_time")), channel.properties.get("wf_start_offset", 0.0))
                   for channel in sources]
        if not np.allclose(rates, rates[0], rtol=1e-10) or any(origin != origins[0] for origin in origins):
            raise MappingRequired("TDMS 通道采样率或时间起点不一致，请明确选择共用时间轴的通道。", options)
        source_units = [channel.properties.get("unit_string", channel.properties.get("unit", "unknown")) for channel in sources]
        if any(unit == "unknown" for unit in source_units) and mapping is None:
            raise MappingRequired("TDMS 缺少通道单位，请明确填写。", {**options, "needs_units": True})
        units = resolve_units(len(sources), mapping, source_units, options=options)
        channels = []
        for channel, unit in zip(sources, units):
            properties = dict(channel.properties)
            declared_scales = properties.get("NI_Number_Of_Scales", 0)
            if declared_scales and properties.get("NI_Scaling_Status", "unscaled") != "scaled":
                from nptdms.scaling import get_scaling
                if get_scaling(properties, contents[channel.group_name].properties, contents.properties) is None:
                    raise ImportError("TDMS 声明了未知缩放链，不能将未缩放整数冒充物理量。")
            channels.append(ChannelInfo(channel.name, unit, "sound_pressure" if unit == "Pa" else "unknown",
                                        scaled=unit in {"V", "Pa", "m/s²", "g"},
                                        metadata={"source_path": channel.path, "source_properties": plain_metadata(properties),
                                                  "source_dtype": str(channel.dtype), "scaling_provider": "nptdms"}))
        return make_signal(path, np.vstack(arrays), rates[0], tuple(channels), mapping=mapping,
                           quality=["unknown_source_unit"] if "unknown" in units else [],
                           metadata={"format": "TDMS", "selected_channels": selected,
                                     "scaling_provider": "nptdms", "source_properties": plain_metadata(dict(contents.properties)),
                                     "source_time_start": origins[0][0], "time_origin_seconds": origins[0][1],
                                     "incomplete_final_segment": False})
    except ImportError:
        raise
    except (ValueError, TypeError, RuntimeError, OSError, KeyError, IndexError) as error:
        raise ImportError(f"无法读取 TDMS：{error}") from error


def import_scientific(path: Path, mapping: ImportMapping | None = None):
    if path.suffix.lower() == ".tdms":
        return import_tdms(path, mapping)
    if h5py.is_hdf5(path):
        return import_hdf5(path, mapping)
    if path.suffix.lower() == ".mat":
        return import_mat(path, mapping)
    raise ImportError("文件不是有效的通用 HDF5；HEAD 文本头须使用专有解析器。")
