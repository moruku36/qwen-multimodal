import pytest

from qmc.gpu_manager import (
    NO_GPU,
    PROFILES,
    GPUInfo,
    MemorySnapshot,
    is_oom_error,
    select_profile,
)


@pytest.mark.parametrize(
    ("name", "gib", "expected"),
    [
        ("NVIDIA A100-SXM4-80GB", 79.2, "a100_80"),
        ("NVIDIA A100-SXM4-40GB", 39.4, "a100_40"),
        ("NVIDIA L4", 22.0, "l4"),
        ("NVIDIA H100 80GB HBM3", 79.1, "a100_80"),
        ("NVIDIA L40S", 44.4, "a100_40"),
        ("Tesla T4", 14.7, "l4"),
    ],
)
def test_select_profile_by_gpu(name, gib, expected):
    assert select_profile(GPUInfo(name=name, total_gib=gib)).key == expected


def test_no_gpu_selects_cpu_profile():
    assert select_profile(NO_GPU).key == "cpu"


def test_override_wins_and_invalid_override_falls_back():
    l4 = GPUInfo("NVIDIA L4", 22.0)
    assert select_profile(l4, override="a100_80").key == "a100_80"
    assert select_profile(l4, override="nope").key == "l4"


def test_mode_labels():
    assert PROFILES["a100_80"].mode == "Performance"
    assert PROFILES["a100_40"].mode == "Performance"
    assert PROFILES["l4"].mode == "Low VRAM"


def test_l4_profile_is_conservative():
    p = PROFILES["l4"]
    assert not p.coresident
    assert p.image_precision == "int8"
    assert p.image_placement == "model_offload"
    assert p.image_max_band <= 1280


def test_is_oom_error():
    class OutOfMemoryError(RuntimeError):
        pass

    assert is_oom_error(OutOfMemoryError("x"))
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not is_oom_error(RuntimeError("file not found"))


def test_snapshot_summary():
    s = MemorySnapshot(
        nvidia_smi_used_gib=10.0, nvidia_smi_total_gib=22.0, torch_allocated_gib=1.0, torch_reserved_gib=2.0
    )
    assert "10.0/22.0" in s.summary()
    assert MemorySnapshot().summary() == "VRAM: n/a"
