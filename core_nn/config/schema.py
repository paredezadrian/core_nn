"""
Configuration schema definitions and validation for CORE-NN.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import yaml
import json
from pathlib import Path


@dataclass
class BCMConfig:
    """Biological Core Memory configuration."""
    memory_size: int = 512
    embedding_dim: int = 768
    salience_threshold: float = 0.7
    decay_rate: float = 0.95
    update_gate_type: str = "gru"
    attention_heads: int = 8


@dataclass
class RTEUConfig:
    """Recursive Temporal Embedding Unit configuration."""
    num_layers: int = 4
    embedding_dim: int = 768
    hidden_dim: int = 2048
    num_capsules: int = 16
    capsule_dim: int = 48
    routing_iterations: int = 3
    temporal_scales: List[int] = field(default_factory=lambda: [1, 4, 16, 64])
    activation: str = "swish"
    dropout: float = 0.1


@dataclass
class IGPMConfig:
    """Instruction-Guided Plasticity Module configuration."""
    plastic_slots: int = 64
    meta_learning_rate: float = 0.001
    fast_weight_decay: float = 0.99
    instruction_embedding_dim: int = 256
    plasticity_threshold: float = 0.8
    max_episodic_memories: int = 1000


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    type: str = "asc"  # Tokenizer type: "asc", "simple", "custom"
    preset: str = "default"  # Preset configuration: "default", "edge", "research"
    custom_config_path: Optional[str] = None  # Path to custom tokenizer config
    overrides: Dict[str, Any] = field(default_factory=dict)  # Override settings


@dataclass
class MLCSConfig:
    """Multi-Level Compression Synthesizer configuration."""
    compression_ratio: float = 0.1
    num_compression_levels: int = 4
    latent_dim: int = 256
    codebook_size: int = 8192
    kpack_max_size_mb: int = 50
    auto_compress_threshold: float = 0.9


@dataclass
class ExecutionEngineConfig:
    """Edge-Efficient Modular Execution Engine configuration."""
    max_concurrent_modules: int = 4
    memory_budget_gb: int = 12
    cpu_threads: int = -1
    offload_threshold: float = 0.8
    async_execution: bool = True
    priority_scheduling: bool = True


@dataclass
class DeviceConfig:
    """Device and performance configuration."""
    preferred: str = "auto"
    mixed_precision: bool = True
    compile_model: bool = False
    memory_efficient: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 1
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 100000


@dataclass
class InferenceConfig:
    """Inference configuration."""
    max_sequence_length: int = 2048
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_new_tokens: int = 512


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    working_memory_size: int = 256
    episodic_memory_size: int = 1024
    semantic_memory_size: int = 4096
    memory_consolidation_interval: int = 100


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    level: str = "INFO"
    log_file: Optional[str] = "core_nn.log"
    log_memory_usage: bool = True
    log_inference_time: bool = True
    tensorboard_dir: Optional[str] = "runs"


@dataclass
class SessionConfig:
    """Session management configuration."""
    auto_save: bool = True
    save_interval: int = 300
    session_dir: str = "sessions"
    max_session_history: int = 10


@dataclass
class APIConfig:
    """API configuration."""
    commands: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "remember": {"enabled": True, "max_items": 100},
        "recall": {"enabled": True, "similarity_threshold": 0.8},
        "forget": {"enabled": True, "confirmation_required": False}
    })


@dataclass
class ModelConfig:
    """Model metadata configuration."""
    name: str = "core-nn-default"
    version: str = "0.2.2"


@dataclass
class CoreNNConfig:
    """Complete CORE-NN configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    bcm: BCMConfig = field(default_factory=BCMConfig)
    rteu: RTEUConfig = field(default_factory=RTEUConfig)
    igpm: IGPMConfig = field(default_factory=IGPMConfig)
    mlcs: MLCSConfig = field(default_factory=MLCSConfig)
    execution_engine: ExecutionEngineConfig = field(default_factory=ExecutionEngineConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    api: APIConfig = field(default_factory=APIConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CoreNNConfig":
        """Create configuration from dictionary."""
        # Create nested configs
        model = ModelConfig(**config_dict.get("model", {}))
        bcm = BCMConfig(**config_dict.get("bcm", {}))
        rteu = RTEUConfig(**config_dict.get("rteu", {}))
        igpm = IGPMConfig(**config_dict.get("igpm", {}))
        mlcs = MLCSConfig(**config_dict.get("mlcs", {}))
        execution_engine = ExecutionEngineConfig(**config_dict.get("execution_engine", {}))
        device = DeviceConfig(**config_dict.get("device", {}))
        training = TrainingConfig(**config_dict.get("training", {}))
        inference = InferenceConfig(**config_dict.get("inference", {}))
        memory = MemoryConfig(**config_dict.get("memory", {}))
        logging = LoggingConfig(**config_dict.get("logging", {}))
        session = SessionConfig(**config_dict.get("session", {}))
        tokenizer = TokenizerConfig(**config_dict.get("tokenizer", {}))
        api = APIConfig(**config_dict.get("api", {}))

        return cls(
            model=model,
            bcm=bcm,
            rteu=rteu,
            igpm=igpm,
            mlcs=mlcs,
            execution_engine=execution_engine,
            device=device,
            training=training,
            inference=inference,
            memory=memory,
            logging=logging,
            session=session,
            tokenizer=tokenizer,
            api=api,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "bcm": self.bcm.__dict__,
            "rteu": self.rteu.__dict__,
            "igpm": self.igpm.__dict__,
            "mlcs": self.mlcs.__dict__,
            "execution_engine": self.execution_engine.__dict__,
            "device": self.device.__dict__,
            "training": self.training.__dict__,
            "inference": self.inference.__dict__,
            "memory": self.memory.__dict__,
            "logging": self.logging.__dict__,
            "session": self.session.__dict__,
            "tokenizer": self.tokenizer.__dict__,
            "api": self.api.__dict__,
        }


def validate_config(config: CoreNNConfig) -> List[str]:
    """Validate configuration and return list of errors."""
    errors = []
    
    # Validate BCM
    if config.bcm.memory_size <= 0:
        errors.append("BCM memory_size must be positive")
    if not 0 < config.bcm.salience_threshold <= 1:
        errors.append("BCM salience_threshold must be between 0 and 1")
    
    # Validate RTEU
    if config.rteu.num_layers <= 0:
        errors.append("RTEU num_layers must be positive")
    if config.rteu.dropout < 0 or config.rteu.dropout >= 1:
        errors.append("RTEU dropout must be between 0 and 1")
    
    # Validate device
    valid_devices = ["auto", "cpu", "cuda", "mps"]
    if config.device.preferred not in valid_devices:
        errors.append(f"Device preferred must be one of {valid_devices}")
    
    return errors
