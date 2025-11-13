import sys
import platform
import psutil

# CONCEPT: What is Personal Information Configuration?
# Developer identity configuration that identifies you as the creator and
# configures your TinyTorch installation. Think Git commit attribution -
# every professional system needs to know who built it.

# CODE STRUCTURE: What We're Building  
def personal_info() -> dict[str, str]:     # Returns developer identity
    return {                               # Dictionary with required fields
        'developer': 'Jeffery_Rain',         # Your actual name
        'email': '22419153@zju.edu.cn',       # Contact information
        'institution': 'Zhejiang University',      # Affiliation
        'system_name': 'Tinytorch_practice',    # Unique system identifier
        'version': '1.0.0'                # Configuration version
    }

# CONNECTIONS: Real-World Equivalents
# Git commits - author name and email in every commit
# Docker images - maintainer information in container metadata
# Python packages - author info in setup.py and pyproject.toml
# Model cards - creator information for ML models

# CONSTRAINTS: Key Implementation Requirements
# - Use actual information (not placeholder text)
# - Email must be valid format (contains @ and domain)
# - System name should be unique and descriptive
# - All values must be strings, version stays '1.0.0'

# CONTEXT: Why This Matters in ML Systems
# Professional ML development requires attribution:
# - Model ownership: Who built this neural network?
# - Collaboration: Others can contact you about issues
# - Professional standards: Industry practice for all software
# - System customization: Makes your TinyTorch installation unique

# CONCEPT: What is System Information?
# Hardware and software environment detection for ML systems.
# Think computer specifications for gaming - ML needs to know what
# resources are available for optimal performance.

# CODE STRUCTURE: What We're Building  
def system_info() -> dict[str, Any]:       # Queries system specs
    
    version_info = sys.version_info
    python_version = f'{version_info.major}.{version_info.minor}.{version_info.micro}'

    platform_name = platform.system()
    architecture = platform.machine()

    cpu_count = psutil.cpu_count()
    memory_bytes = psutil.virtual_memory().total
    memory_gb = round(memory_bytes / (1024**3), 1)

    return { # Hardware/software details
        'python_version': f'{python_version}',      # Python compatibility
        'platform':f'{platform_name}',              # Operating system
        'architecture': f'{architecture}',          # CPU architecture
        'cpu_count': f'{cpu_count}',                # Parallel processing cores
        'memory_gb': f'{memory_gb}'                 # Available RAM
    }

# CONNECTIONS: Real-World Equivalents
# torch.get_num_threads() (PyTorch) - uses CPU count for optimization
# tf.config.list_physical_devices() (TensorFlow) - queries hardware
# psutil.cpu_count() (System monitoring) - same underlying queries
# MLflow system tracking - documents environment for reproducibility

# CONSTRAINTS: Key Implementation Requirements
# - Use actual system queries (not hardcoded values)
# - Convert memory from bytes to GB for readability
# - Round memory to 1 decimal place for clean output
# - Return proper data types (strings, int, float)

# CONTEXT: Why This Matters in ML Systems
# Hardware awareness enables performance optimization:
# - Training: More CPU cores = faster data processing
# - Memory: Determines maximum model and batch sizes
# - Debugging: System specs help troubleshoot performance issues
# - Reproducibility: Document exact environment for experiment tracking