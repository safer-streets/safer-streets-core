"""
Environment configuration, loaded with pydantic-settings.

Values come from environment variables (matched case-insensitively on the field name), falling back to the nearest ``.env``
file found searching upwards from the working directory; real environment variables take precedence over ``.env``.

Settings are split by concern so that each only demands the variables it needs: constructing one with a required
variable unset raises a ``ValidationError`` naming it. They are deliberately constructed per use rather than cached at
import, so that importing the package never needs credentials and tests can patch the environment.
"""

from pathlib import Path

from dotenv import find_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class _EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore")

    def __init__(self) -> None:
        # located per construction, not at import, so that it follows the working directory
        super().__init__(_env_file=find_dotenv(usecwd=True) or None)


class LocalDataSettings(_EnvSettings):
    safer_streets_data_dir: Path


class BlobStorageSettings(_EnvSettings):
    safer_streets_blob_storage: str


class AzureSettings(_EnvSettings):
    azure_storage_connstr: SecretStr


class AzureAdminSettings(_EnvSettings):
    azure_storage_admin_connstr: SecretStr


class NomisSettings(_EnvSettings):
    nomis_api_key: SecretStr


class CartoSettings(_EnvSettings):
    carto_api_key: SecretStr | None = None
