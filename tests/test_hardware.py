from omnillm.core.hardware import (
    HardwareProfile,
    QuantFitStatus,
    detect_hardware,
    estimate_model_memory,
    parse_model_parameter_count,
    recommend_model_fit,
    suggest_models_for_hardware,
)


def test_detect_hardware() -> None:
    hw = detect_hardware()
    assert isinstance(hw, HardwareProfile)
    assert hw.cpu_count >= 1
    assert hw.total_ram_gb > 0
    assert hw.available_ram_gb >= 0
    assert isinstance(hw.to_dict(), dict)
    assert len(hw.summary()) > 0


def test_parse_model_parameter_count() -> None:
    assert parse_model_parameter_count("llama-3.1-8b") == 8.0
    assert parse_model_parameter_count("llama3:8b") == 8.0
    assert parse_model_parameter_count("llama-3.2-1b") == 1.0
    assert parse_model_parameter_count("llama-3.2-3b") == 3.0
    assert parse_model_parameter_count("llama-3.3-70b") == 70.0
    assert parse_model_parameter_count("phi-3-mini") == 3.8
    assert parse_model_parameter_count("mistral:7b") == 7.0
    assert parse_model_parameter_count("smollm:135m") == 0.135
    assert parse_model_parameter_count("smollm2:360m") == 0.36
    assert parse_model_parameter_count("mixtral-8x7b") is not None
    assert parse_model_parameter_count("unknown-random-model-xyz") is None


def test_estimate_model_memory() -> None:
    # 8B model memory estimation
    total_fp16, w_fp16, kv_fp16 = estimate_model_memory(8.0, quant="FP16", context_length=4096)
    total_q8, w_q8, kv_q8 = estimate_model_memory(8.0, quant="Q8_0", context_length=4096)
    total_q4, w_q4, kv_q4 = estimate_model_memory(8.0, quant="Q4_K_M", context_length=4096)
    total_q2, w_q2, kv_q2 = estimate_model_memory(8.0, quant="Q2_K", context_length=4096)

    # Weights scale monotonically with quantization bitwidth
    assert w_fp16 > w_q8 > w_q4 > w_q2
    assert total_fp16 > total_q8 > total_q4 > total_q2

    # Higher context window increases KV cache
    total_4k, _, kv_4k = estimate_model_memory(8.0, quant="Q4_K_M", context_length=4096)
    total_8k, _, kv_8k = estimate_model_memory(8.0, quant="Q4_K_M", context_length=8192)
    assert kv_8k > kv_4k
    assert total_8k > total_4k


def test_recommend_model_fit_cuda_high_vram() -> None:
    # Mock RTX 4090 with 24 GB VRAM
    hw = HardwareProfile(
        os_name="Linux",
        os_release="6.8.0",
        architecture="x86_64",
        cpu_count=16,
        total_ram_gb=64.0,
        available_ram_gb=48.0,
        gpu_name="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        total_vram_gb=24.0,
        free_vram_gb=23.0,
        has_cuda=True,
        is_apple_silicon=False,
    )

    rec = recommend_model_fit("llama-3.1-8b", context_length=4096, hardware=hw)
    assert rec.can_fit is True
    assert rec.best_target == QuantFitStatus.GPU_FULL
    assert rec.recommended_quant in ("Q8_0", "Q5_K_M", "Q4_K_M")
    assert "RTX 4090" in rec.format_table()


def test_recommend_model_fit_cuda_low_vram() -> None:
    # Mock GTX 1650 with 4 GB VRAM and 16 GB RAM
    hw = HardwareProfile(
        os_name="Linux",
        os_release="6.8.0",
        architecture="x86_64",
        cpu_count=8,
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        gpu_name="NVIDIA GeForce GTX 1650",
        gpu_count=1,
        total_vram_gb=4.0,
        free_vram_gb=3.5,
        has_cuda=True,
        is_apple_silicon=False,
    )

    # 3B model fits in GPU VRAM
    rec3b = recommend_model_fit("llama-3.2-3b", hardware=hw)
    assert rec3b.can_fit is True
    assert rec3b.best_target == QuantFitStatus.GPU_FULL

    # 8B model cannot fit in 4GB GPU VRAM, offloads to CPU RAM
    rec8b = recommend_model_fit("llama-3.1-8b", hardware=hw)
    assert rec8b.can_fit is True
    assert rec8b.best_target == QuantFitStatus.CPU_RAM
    assert rec8b.recommended_quant == "Q4_K_M"


def test_recommend_model_fit_apple_silicon() -> None:
    # Mock Apple M3 Max with 64 GB Unified Memory
    hw = HardwareProfile(
        os_name="Darwin",
        os_release="23.4.0",
        architecture="arm64",
        cpu_count=14,
        total_ram_gb=64.0,
        available_ram_gb=50.0,
        gpu_name="Apple Silicon (Apple M3 Max)",
        gpu_count=1,
        total_vram_gb=48.0,
        free_vram_gb=37.5,
        has_cuda=False,
        is_apple_silicon=True,
    )

    rec70b = recommend_model_fit("llama-3.3-70b", hardware=hw)
    assert rec70b.can_fit is True
    assert rec70b.best_target == QuantFitStatus.METAL_UNIFIED
    assert rec70b.recommended_quant == "Q4_K_M"


def test_recommend_model_fit_oom() -> None:
    # Mock low-spec machine with 8 GB RAM and no GPU
    hw = HardwareProfile(
        os_name="Linux",
        os_release="6.8.0",
        architecture="x86_64",
        cpu_count=4,
        total_ram_gb=8.0,
        available_ram_gb=4.0,
        has_cuda=False,
        is_apple_silicon=False,
    )

    rec70b = recommend_model_fit("llama-3.3-70b", hardware=hw)
    assert rec70b.can_fit is False
    assert rec70b.best_target == QuantFitStatus.OOM


def test_suggest_models_for_hardware() -> None:
    hw = HardwareProfile(
        os_name="Linux",
        os_release="6.8.0",
        architecture="x86_64",
        cpu_count=8,
        total_ram_gb=16.0,
        available_ram_gb=10.0,
        has_cuda=False,
        is_apple_silicon=False,
    )

    suggestions = suggest_models_for_hardware(hw)
    assert len(suggestions) > 0
    # 135m should fit easily on 16GB RAM
    smollm = next(s for s in suggestions if "135m" in s["model"])
    assert smollm["can_fit"] is True
    # 70B should not fit on 16GB RAM
    llama70b = next(s for s in suggestions if "70b" in s["model"])
    assert llama70b["can_fit"] is False
