# backend/tests/test_config.py
import os
from pathlib import Path

def test_config_loads_from_env(monkeypatch, tmp_path):
    # Simulate a .env file location
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-proj-UNITTEST\nFILTER_BLOCK=0.9\nCORS_ORIGINS=http://localhost:5173,https://example.com\n")

    # Ensure python-dotenv will pick it up by being CWD
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Import after CWD change so dotenv finds .env
        from importlib import reload
        import config  # backend/config.py
        reload(config)  # re-run dotenv + build settings

        # Validate parsed values
        s = config.settings
        assert s.openai_api_key == "sk-proj-UNITTEST"
        assert s.filter_block == 0.9
        assert s.cors_origins == ["http://localhost:5173", "https://example.com"]
    finally:
        os.chdir(cwd)

def test_config_env_var_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-OVERRIDE")
    monkeypatch.setenv("FILTER_ALLOW", "0.02")
    monkeypatch.setenv("CORS_ORIGINS", "*,https://prod.example.com")

    from importlib import reload
    import config
    reload(config)
    s = config.settings
    assert s.openai_api_key == "sk-proj-OVERRIDE"
    assert s.filter_allow == 0.02
    assert s.cors_origins == ["*", "https://prod.example.com"]
