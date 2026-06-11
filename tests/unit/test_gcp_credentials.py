from unittest.mock import patch

import pytest

from agent.gcp_credentials import load_gcp_credentials, resolve_credentials_path


def test_resolve_credentials_path_prefers_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/env/creds.json")
    assert resolve_credentials_path(settings_path="/settings/creds.json") == "/settings/creds.json"
    assert resolve_credentials_path(settings_path="") == "/env/creds.json"


def test_load_gcp_credentials_from_service_account_file(tmp_path) -> None:
    creds_file = tmp_path / "sa.json"
    creds_file.write_text(
        '{"type":"service_account","project_id":"exaai-sdk",'
        '"private_key_id":"x","private_key":"-----BEGIN PRIVATE KEY-----\\n'
        'MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7\\n'
        '-----END PRIVATE KEY-----\\n",'
        '"client_email":"sa@exaai-sdk.iam.gserviceaccount.com",'
        '"client_id":"1","auth_uri":"https://accounts.google.com/o/oauth2/auth",'
        '"token_uri":"https://oauth2.googleapis.com/token"}',
        encoding="utf-8",
    )
    with patch("google.oauth2.service_account.Credentials.from_service_account_file") as mock_load:
        mock_load.return_value.valid = True
        mock_load.return_value.token = "token"
        load_gcp_credentials(settings_path=str(creds_file))
        mock_load.assert_called_once()
