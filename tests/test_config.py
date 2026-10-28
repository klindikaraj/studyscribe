from pathlib import Path

from studyscribe.config import APP_DIR_NAME, DATA_DIR_ENV, default_data_dir, resolve_data_dir


def test_default_is_under_localappdata(tmp_path: Path) -> None:
    env = {"LOCALAPPDATA": str(tmp_path)}
    assert resolve_data_dir(None, env) == (tmp_path / APP_DIR_NAME).resolve()


def test_env_var_overrides_default(tmp_path: Path) -> None:
    custom = tmp_path / "custom"
    env = {"LOCALAPPDATA": str(tmp_path), DATA_DIR_ENV: str(custom)}
    assert resolve_data_dir(None, env) == custom.resolve()


def test_cli_value_overrides_env_var(tmp_path: Path) -> None:
    cli = tmp_path / "from-cli"
    env = {"LOCALAPPDATA": str(tmp_path), DATA_DIR_ENV: str(tmp_path / "from-env")}
    assert resolve_data_dir(cli, env) == cli.resolve()


def test_empty_values_fall_through(tmp_path: Path) -> None:
    env = {"LOCALAPPDATA": str(tmp_path), DATA_DIR_ENV: ""}
    assert resolve_data_dir("", env) == (tmp_path / APP_DIR_NAME).resolve()


def test_default_does_not_depend_on_cwd(tmp_path: Path, monkeypatch) -> None:
    env = {"LOCALAPPDATA": str(tmp_path / "appdata")}
    before = default_data_dir(env)
    monkeypatch.chdir(tmp_path)
    assert default_data_dir(env) == before
