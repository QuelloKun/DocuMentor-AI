"""
Test file to verify the setup and environment configuration.
"""

import pytest
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_python_version():
    """Test that we're using Python 3.8+"""
    assert sys.version_info >= (3, 8), f"Python version {sys.version_info} is too old"


def test_core_imports():
    """Test that core libraries can be imported"""
    try:
        import pandas
        import numpy
        import arxiv
        import jupyter
        import tqdm
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import core library: {e}")


def test_config_import():
    """Test that our config file can be imported"""
    try:
        import config
        assert hasattr(config, 'PROJECT_ROOT')
        assert hasattr(config, 'ARXIV_CONFIG')
        assert hasattr(config, 'MODEL_CONFIG')
    except ImportError as e:
        pytest.fail(f"Failed to import config: {e}")


def test_directory_structure():
    """Test that the required directories exist"""
    project_root = Path(__file__).parent.parent
    required_dirs = [
        "src", "src/data", "src/models", "src/api", "src/ui", "src/utils",
        "data", "data/raw", "data/processed", "data/embeddings",
        "notebooks", "tests", "docs"
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Required directory {dir_name} does not exist"


def test_git_files():
    """Test that Git configuration files exist"""
    project_root = Path(__file__).parent.parent
    git_files = [".gitignore", ".gitattributes"]
    
    for file_name in git_files:
        file_path = project_root / file_name
        assert file_path.exists(), f"Required Git file {file_name} does not exist"


def test_project_files():
    """Test that essential project files exist"""
    project_root = Path(__file__).parent.parent
    project_files = ["README.md", "requirements.txt", "config.py", "Project Plan.md"]
    
    for file_name in project_files:
        file_path = project_root / file_name
        assert file_path.exists(), f"Required project file {file_name} does not exist"


if __name__ == "__main__":
    pytest.main([__file__]) 