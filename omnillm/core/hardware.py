import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class QuantFitStatus(StrEnum):
    GPU_FULL = "GPU (Full Offload)"
    METAL_UNIFIED = "Metal Unified Memory"
    CPU_RAM = "CPU / System RAM"
    OOM = "Insufficient Memory"


QUANT_SPECS: dict[str, dict[str, Any]] = {
    "FP16": {
        "bytes_per_param": 2.00,
        "quality": "Maximum (Lossless)",
        "speed": "Fast on high-end GPUs",
    },
    "Q8_0": {
        "bytes_per_param": 1.05,
        "quality": "Near Lossless (~99.5%)",
        "speed": "Fast (Minimal overhead)",
    },
    "Q6_K": {
        "bytes_per_param": 0.82,
        "quality": "Extremely High (>99%)",
        "speed": "Fast",
    },
    "Q5_K_M": {
        "bytes_per_param": 0.70,
        "quality": "High Sweet-Spot (~98%)",
        "speed": "Fast / Highly Balanced",
    },
    "Q4_K_M": {
        "bytes_per_param": 0.58,
        "quality": "Recommended Standard (~95%)",
        "speed": "Fastest / Standard",
    },
    "Q3_K_M": {
        "bytes_per_param": 0.48,
        "quality": "Noticeable Degradation",
        "speed": "Fast, Smallest Footprint",
    },
    "Q2_K": {
        "bytes_per_param": 0.38,
        "quality": "High Perplexity Loss",
        "speed": "Minimal Memory Required",
    },
}

KNOWN_MODEL_SIZES: dict[str, float] = {
    "llama3": 8.0,
    "llama-3": 8.0,
    "llama3.1": 8.0,
    "llama-3.1": 8.0,
    "llama3.2": 3.0,
    "llama-3.2": 3.0,
    "llama3.3": 70.0,
    "llama-3.3": 70.0,
    "mistral": 7.0,
    "mistral-nemo": 12.0,
    "mixtral-8x7b": 46.7,
    "mixtral": 46.7,
    "phi3": 3.8,
    "phi-3": 3.8,
    "phi-3-mini": 3.8,
    "phi4": 14.7,
    "phi-4": 14.7,
    "gemma": 7.0,
    "gemma2": 9.0,
    "gemma-2": 9.0,
    "deepseek-r1": 7.0,
    "qwen2": 7.0,
    "qwen2.5": 7.0,
}


@dataclass
class HardwareProfile:
    os_name: str
    os_release: str
    architecture: str
    cpu_count: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_name: str | None = None
    gpu_count: int = 0
    total_vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    is_apple_silicon: bool = False
    has_cuda: bool = False

    def primary_memory_pool_gb(self) -> float:
        """Returns the primary memory pool (in GB) available for model acceleration."""
        if self.is_apple_silicon:
            return self.total_ram_gb * 0.75
        if self.has_cuda and self.total_vram_gb > 0:
            return self.total_vram_gb
        return self.available_ram_gb

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        parts = [
            f"OS: {self.os_name} {self.os_release} ({self.architecture}, {self.cpu_count} CPUs)",
            f"RAM: {self.total_ram_gb:.2f} GB Total ({self.available_ram_gb:.2f} GB Available)",
        ]
        if self.has_cuda and self.gpu_name:
            parts.append(
                f"GPU: {self.gpu_name} ({self.total_vram_gb:.2f} GB Total VRAM, {self.free_vram_gb:.2f} GB Free)"
            )
        elif self.is_apple_silicon:
            parts.append(f"GPU: {self.gpu_name or 'Apple Silicon Metal'} (Unified Memory Architecture)")
        else:
            parts.append("Accelerator: CPU Inference Only")
        return " | ".join(parts)


@dataclass
class QuantFitResult:
    quant: str
    required_memory_gb: float
    weights_memory_gb: float
    kv_cache_memory_gb: float
    status: QuantFitStatus
    fits: bool
    quality_description: str
    speed_description: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class FitRecommendation:
    model_name: str
    param_count_b: float
    context_length: int
    hardware: HardwareProfile
    results: list[QuantFitResult]
    can_fit: bool
    recommended_quant: str | None
    best_target: QuantFitStatus
    summary_message: str
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "param_count_b": self.param_count_b,
            "context_length": self.context_length,
            "hardware": self.hardware.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "can_fit": self.can_fit,
            "recommended_quant": self.recommended_quant,
            "best_target": self.best_target.value,
            "summary_message": self.summary_message,
            "suggestions": self.suggestions,
        }

    def format_table(self) -> str:
        lines: list[str] = []
        lines.append("🖥️  Detected Hardware Profile")
        lines.append("-" * 78)
        lines.append(
            f"OS:           {self.hardware.os_name} {self.hardware.os_release} ({self.hardware.architecture}, {self.hardware.cpu_count} CPUs)"
        )
        lines.append(
            f"System RAM:   {self.hardware.total_ram_gb:.2f} GB Total ({self.hardware.available_ram_gb:.2f} GB Available)"
        )
        if self.hardware.has_cuda and self.hardware.gpu_name:
            lines.append(
                f"GPU:          {self.hardware.gpu_name} ({self.hardware.total_vram_gb:.2f} GB VRAM, {self.hardware.free_vram_gb:.2f} GB Free)"
            )
        elif self.hardware.is_apple_silicon:
            lines.append(f"GPU:          {self.hardware.gpu_name or 'Apple Silicon Metal'} (Unified Memory)")
        else:
            lines.append("Accelerator:  CPU Only (No dedicated GPU detected)")
        lines.append("")

        lines.append(
            f"🧠 Model Fit Analysis: {self.model_name} (~{self.param_count_b:.1f}B params, {self.context_length:,} context)"
        )
        lines.append("-" * 78)
        lines.append(f"{'QUANT':<9} {'SIZE (GB)':<11} {'FIT STATUS':<24} {'QUALITY':<22} {'SPEED'}")
        lines.append("-" * 78)

        for r in self.results:
            icon = "✅" if r.fits and r.status != QuantFitStatus.CPU_RAM else ("⚠️" if r.fits else "❌")
            status_text = f"{icon} {r.status.value}"
            speed_text = r.speed_description if r.fits else "OOM"
            lines.append(
                f"{r.quant:<9} {r.required_memory_gb:>5.2f} GB    {status_text:<24} {r.quality_description:<22} {speed_text}"
            )

        lines.append("-" * 78)
        lines.append("\n💡 Recommendation:")
        lines.append(f"   {self.summary_message}")
        for s in self.suggestions:
            lines.append(f"   {s}")
        return "\n".join(lines)


def detect_hardware() -> HardwareProfile:
    """Detects host operating system, CPU architecture, system RAM, and GPU accelerators (NVIDIA/Apple Silicon)."""
    os_name = platform.system()
    os_release = platform.release()
    arch = platform.machine()
    cpu_count = os.cpu_count() or 1

    total_ram_gb = 8.0
    available_ram_gb = 4.0

    # 1. Try psutil if available
    try:
        import psutil  # type: ignore[import-untyped]

        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / (1024**3)
        available_ram_gb = vm.available / (1024**3)
    except Exception:
        # Fallback 1: Linux /proc/meminfo
        if os_name == "Linux" and os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_ram_gb = int(line.split()[1]) / (1024 * 1024)
                        elif line.startswith("MemAvailable:"):
                            available_ram_gb = int(line.split()[1]) / (1024 * 1024)
            except Exception:
                pass
        # Fallback 2: macOS sysctl
        elif os_name == "Darwin":
            try:
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2).strip()
                total_ram_gb = int(out) / (1024**3)
                available_ram_gb = total_ram_gb * 0.70  # safe heuristic
            except Exception:
                pass
        # Fallback 3: Windows GlobalMemoryStatusEx
        elif sys.platform == "win32":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
                total_ram_gb = stat.ullTotalPhys / (1024**3)
                available_ram_gb = stat.ullAvailPhys / (1024**3)
            except Exception:
                pass

    # 2. GPU Detection
    gpu_name: str | None = None
    gpu_count = 0
    total_vram_gb = 0.0
    free_vram_gb = 0.0
    has_cuda = False
    is_apple_silicon = False

    # Check NVIDIA via nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.check_output(
                [nvidia_smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                text=True,
                timeout=3,
            ).strip()
            lines = [line.strip() for line in out.splitlines() if line.strip()]
            if lines:
                gpu_count = len(lines)
                has_cuda = True
                first_parts = [p.strip() for p in lines[0].split(",")]
                gpu_name = first_parts[0]
                total_vram_gb = float(first_parts[1]) / 1024.0
                free_vram_gb = float(first_parts[2]) / 1024.0
        except Exception:
            pass

    # Check Apple Silicon
    if os_name == "Darwin" and arch in ("arm64", "aarch64"):
        is_apple_silicon = True
        try:
            chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2).strip()
            gpu_name = f"Apple Silicon ({chip})"
        except Exception:
            gpu_name = "Apple Silicon GPU (Metal)"
        total_vram_gb = total_ram_gb * 0.75
        free_vram_gb = available_ram_gb * 0.75

    return HardwareProfile(
        os_name=os_name,
        os_release=os_release,
        architecture=arch,
        cpu_count=cpu_count,
        total_ram_gb=round(total_ram_gb, 2),
        available_ram_gb=round(available_ram_gb, 2),
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        total_vram_gb=round(total_vram_gb, 2),
        free_vram_gb=round(free_vram_gb, 2),
        is_apple_silicon=is_apple_silicon,
        has_cuda=has_cuda,
    )


def parse_model_parameter_count(model_name: str) -> float | None:
    """Infers the parameter count in billions from common model names or aliases."""
    name = model_name.lower().strip()

    # Explicit billion patterns: e.g. "8b", "70b", "3.8b", "0.5b"
    b_match = re.search(r"(?:^|[-_:a-zA-Z])(\d+(?:\.\d+)?)[bB](?:[-_:]|$)", name)
    if b_match:
        return float(b_match.group(1))

    # Explicit million patterns: e.g. "135m", "360m"
    m_match = re.search(r"(?:^|[-_:a-zA-Z])(\d+(?:\.\d+)?)[mM](?:[-_:]|$)", name)
    if m_match:
        return float(m_match.group(1)) / 1000.0

    # Explicit MoE patterns: e.g. "8x7b"
    moe_match = re.search(r"(\d+)x(\d+(?:\.\d+)?)[bB]", name)
    if moe_match:
        count = int(moe_match.group(1))
        size = float(moe_match.group(2))
        return count * size * 0.83  # Shared embedding and active expert factor

    # Fallback to known lookup
    for key, size in KNOWN_MODEL_SIZES.items():
        if key in name:
            return size

    return None


def estimate_model_memory(
    param_count_b: float,
    quant: str = "Q4_K_M",
    context_length: int = 4096,
) -> tuple[float, float, float]:
    """Estimates the required memory (total, weights, kv_cache) in GB for a given model, quantization, and context."""
    spec = QUANT_SPECS.get(quant.upper(), QUANT_SPECS["Q4_K_M"])
    bytes_per_param = spec["bytes_per_param"]

    # 1. Weights memory (GB)
    weights_gb = param_count_b * bytes_per_param

    # 2. KV Cache memory approximation (GB)
    # Scaled by parameter size & context window
    kv_base_factor = 0.5 * (param_count_b / 8.0)
    kv_cache_gb = max(0.15, kv_base_factor * (context_length / 4096.0))

    # 3. Runtime activation & CUDA context overhead
    runtime_overhead_gb = 0.40

    total_gb = weights_gb + kv_cache_gb + runtime_overhead_gb
    return round(total_gb, 2), round(weights_gb, 2), round(kv_cache_gb, 2)


def recommend_model_fit(
    model_name: str,
    context_length: int = 4096,
    param_count_b: float | None = None,
    hardware: HardwareProfile | None = None,
) -> FitRecommendation:
    """Evaluates whether a model can fit in available hardware and recommends optimal quantization."""
    if hardware is None:
        hardware = detect_hardware()

    if param_count_b is None:
        param_count_b = parse_model_parameter_count(model_name)
        if param_count_b is None:
            # Default fallback assumption
            param_count_b = 7.0

    results: list[QuantFitResult] = []
    recommended_quant: str | None = None
    best_target: QuantFitStatus = QuantFitStatus.OOM

    # Available pools
    vram_pool = hardware.total_vram_gb if hardware.has_cuda else 0.0
    metal_pool = (hardware.total_ram_gb * 0.75) if hardware.is_apple_silicon else 0.0
    system_pool = hardware.available_ram_gb

    for quant_name, spec in QUANT_SPECS.items():
        total_gb, weights_gb, kv_gb = estimate_model_memory(param_count_b, quant_name, context_length)

        if hardware.has_cuda and total_gb <= (vram_pool * 0.95):
            status = QuantFitStatus.GPU_FULL
            fits = True
            speed_note = "Fast (Full GPU acceleration)"
            notes = "Fits entirely in GPU VRAM."
        elif hardware.is_apple_silicon and total_gb <= metal_pool:
            status = QuantFitStatus.METAL_UNIFIED
            fits = True
            speed_note = "Fast (Metal Unified GPU)"
            notes = "Fits in Apple Silicon Metal unified memory."
        elif total_gb <= (system_pool * 0.90) or total_gb <= (hardware.total_ram_gb * 0.85):
            status = QuantFitStatus.CPU_RAM
            fits = True
            speed_note = "Moderate / Slow (CPU Compute)"
            notes = "Will run on CPU RAM (slower than GPU offload)."
        else:
            status = QuantFitStatus.OOM
            fits = False
            speed_note = "OOM"
            notes = "Exceeds available memory."

        results.append(
            QuantFitResult(
                quant=quant_name,
                required_memory_gb=total_gb,
                weights_memory_gb=weights_gb,
                kv_cache_memory_gb=kv_gb,
                status=status,
                fits=fits,
                quality_description=spec["quality"],
                speed_description=speed_note,
                notes=notes,
            )
        )

    # Determine recommendation
    # Priority: Find highest quality quantization that fits on GPU/Metal, or standard Q4_K_M on CPU
    fitting_results = [r for r in results if r.fits]
    suggestions: list[str] = []

    if fitting_results:
        # Check if any fits on GPU / Metal
        accelerated = [
            r for r in fitting_results if r.status in (QuantFitStatus.GPU_FULL, QuantFitStatus.METAL_UNIFIED)
        ]
        if accelerated:
            # Prefer Q5_K_M or Q4_K_M if possible, or Q8_0 if plenty of space
            preferred = next((r for r in accelerated if r.quant in ("Q5_K_M", "Q4_K_M", "Q8_0")), accelerated[0])
            recommended_quant = preferred.quant
            best_target = preferred.status
            summary_message = (
                f"Recommended: {recommended_quant} on {best_target.value} (~{preferred.required_memory_gb:.2f} GB). "
                f"Fits entirely with high performance!"
            )
        else:
            # CPU only
            q4 = next((r for r in fitting_results if r.quant == "Q4_K_M"), fitting_results[-1])
            recommended_quant = q4.quant
            best_target = QuantFitStatus.CPU_RAM
            summary_message = (
                f"Recommended: {recommended_quant} on {best_target.value} (~{q4.required_memory_gb:.2f} GB). "
                f"Exceeds GPU VRAM; will run using CPU RAM."
            )
            if hardware.has_cuda and hardware.total_vram_gb > 0:
                suggestions.append(
                    f"💡 Tip: For full GPU acceleration, try a smaller model (e.g., 3B or 1B) to fit within your {hardware.total_vram_gb:.1f} GB VRAM."
                )
    else:
        can_fit = False
        summary_message = f"Model {model_name} (~{param_count_b:.1f}B) is too large for your available hardware ({hardware.total_ram_gb:.1f} GB RAM)."
        suggestions.append("💡 Tip: Consider using a smaller model (e.g. 7B, 3B, or 1B) or cloud offload.")

    can_fit = len(fitting_results) > 0

    return FitRecommendation(
        model_name=model_name,
        param_count_b=param_count_b,
        context_length=context_length,
        hardware=hardware,
        results=results,
        can_fit=can_fit,
        recommended_quant=recommended_quant,
        best_target=best_target,
        summary_message=summary_message,
        suggestions=suggestions,
    )


def suggest_models_for_hardware(hardware: HardwareProfile | None = None) -> list[dict[str, Any]]:
    """Generates a list of popular local models categorized by how well they run on the detected hardware."""
    if hardware is None:
        hardware = detect_hardware()

    candidate_models = [
        ("smollm:135m", 0.135, "Ultra-fast lightweight model for embedded & testing"),
        ("llama-3.2:1b", 1.0, "High-efficiency edge & mobile reasoning"),
        ("llama-3.2:3b", 3.0, "Exceptional quality-to-size ratio for laptops & consumer GPUs"),
        ("phi-3-mini:3.8b", 3.8, "State of the art reasoning on consumer hardware"),
        ("mistral:7b", 7.0, "Industry standard general-purpose workhorse"),
        ("llama-3.1:8b", 8.0, "Top-tier 8B model with strong coding and instruction following"),
        ("deepseek-r1:14b", 14.0, "Advanced mathematical and reasoning capability"),
        ("qwen2.5:32b", 32.0, "Near-frontier performance for workstations"),
        ("llama-3.3:70b", 70.0, "Frontier-level open model for multi-GPU servers"),
    ]

    suggestions = []
    for name, size, desc in candidate_models:
        fit = recommend_model_fit(name, context_length=4096, param_count_b=size, hardware=hardware)
        suggestions.append(
            {
                "model": name,
                "params": f"{size}B",
                "can_fit": fit.can_fit,
                "target": fit.best_target.value,
                "recommended_quant": fit.recommended_quant,
                "required_gb": next(
                    (r.required_memory_gb for r in fit.results if r.quant == fit.recommended_quant), 0.0
                )
                if fit.recommended_quant
                else 0.0,
                "description": desc,
            }
        )

    return suggestions
