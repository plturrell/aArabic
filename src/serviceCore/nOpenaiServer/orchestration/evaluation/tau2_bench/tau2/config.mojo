# tau2/config.mojo
# Migrated from tau2/config.py
# Configuration constants for TAU2-Bench evaluation framework

# SIMULATION
alias DEFAULT_MAX_STEPS: Int = 200
alias DEFAULT_MAX_ERRORS: Int = 10
alias DEFAULT_SEED: Int = 300
alias DEFAULT_MAX_CONCURRENCY: Int = 64
alias DEFAULT_NUM_TRIALS: Int = 1
alias DEFAULT_LOG_LEVEL: String = "INFO"

# LLM - Note: Using local LLM from /Users/user/Documents/arabic_folder/src/serviceCore/nOpenaiServer
alias DEFAULT_AGENT_IMPLEMENTATION: String = "llm_agent"
alias DEFAULT_USER_IMPLEMENTATION: String = "user_simulator"
alias DEFAULT_LLM_AGENT: String = "local-model"
alias DEFAULT_LLM_USER: String = "local-model"
alias DEFAULT_LLM_TEMPERATURE_AGENT: Float64 = 1.0
alias DEFAULT_LLM_TEMPERATURE_USER: Float64 = 1.0

alias DEFAULT_LLM_NL_ASSERTIONS: String = "local-model"
alias DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE: Float64 = 1.0

alias DEFAULT_LLM_ENV_INTERFACE: String = "local-model"
alias DEFAULT_LLM_ENV_INTERFACE_TEMPERATURE: Float64 = 1.0

# LITELLM
alias DEFAULT_MAX_RETRIES: Int = 3
alias LLM_CACHE_ENABLED: Bool = False
alias DEFAULT_LLM_CACHE_TYPE: String = "redis"

# REDIS CACHE
alias REDIS_HOST: String = "localhost"
alias REDIS_PORT: Int = 6379
alias REDIS_PASSWORD: String = ""
alias REDIS_PREFIX: String = "tau2"
alias REDIS_CACHE_VERSION: String = "v1"
alias REDIS_CACHE_TTL: Int = 60 * 60 * 24 * 30  # 30 days

# LANGFUSE
alias USE_LANGFUSE: Bool = False

# API - Aligned with toon_http_service architecture
alias API_PORT: Int = 8000  # TAU2-Bench API server
alias TOON_SERVICE_PORT: Int = 8003  # TOON encoding service
alias TOON_SERVICE_URL: String = "http://localhost:8003"

# Service Integration
alias USE_TOON_ENCODING: Bool = False  # Enable TOON format for data efficiency

# LLM Arguments structure
struct LLMArgs:
    var temperature: Float64
    
    fn __init__(inout self, temperature: Float64 = 1.0):
        self.temperature = temperature

fn get_default_llm_args_agent() -> LLMArgs:
    return LLMArgs(temperature=DEFAULT_LLM_TEMPERATURE_AGENT)

fn get_default_llm_args_user() -> LLMArgs:
    return LLMArgs(temperature=DEFAULT_LLM_TEMPERATURE_USER)

fn get_default_llm_args_nl_assertions() -> LLMArgs:
    return LLMArgs(temperature=DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE)

fn get_default_llm_args_env_interface() -> LLMArgs:
    return LLMArgs(temperature=DEFAULT_LLM_ENV_INTERFACE_TEMPERATURE)
