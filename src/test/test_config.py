import pytest
from pydantic import ValidationError

from safer_streets_core.config import CartoSettings, NomisSettings


@pytest.fixture
def no_nomis_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOMIS_API_KEY", raising=False)


def test_reads_env_file_in_parent_folder(tmp_path, monkeypatch, no_nomis_key):
    (tmp_path / ".env").write_text("NOMIS_API_KEY=from-parent\n")
    subdir = tmp_path / "a" / "b"
    subdir.mkdir(parents=True)
    monkeypatch.chdir(subdir)
    assert NomisSettings().nomis_api_key.get_secret_value() == "from-parent"


def test_nearest_env_file_wins(tmp_path, monkeypatch, no_nomis_key):
    (tmp_path / ".env").write_text("NOMIS_API_KEY=outer\n")
    subdir = tmp_path / "inner"
    subdir.mkdir()
    (subdir / ".env").write_text("NOMIS_API_KEY=inner\n")
    monkeypatch.chdir(subdir)
    assert NomisSettings().nomis_api_key.get_secret_value() == "inner"


def test_environment_overrides_env_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("NOMIS_API_KEY=from-file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NOMIS_API_KEY", "from-env")
    assert NomisSettings().nomis_api_key.get_secret_value() == "from-env"


def test_missing_required_variable_raises(tmp_path, monkeypatch, no_nomis_key):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError, match="nomis_api_key"):
        NomisSettings()


def test_optional_variable_defaults_to_none(tmp_path, monkeypatch):
    monkeypatch.delenv("CARTO_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    assert CartoSettings().carto_api_key is None
