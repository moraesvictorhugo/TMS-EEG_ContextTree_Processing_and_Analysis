import os


def setup_plotting_backend():
    """Select a plotting backend based on the execution environment."""
    import matplotlib
    import mne

    # Detect if it's remote access (SSH, WSL, etc.)
    is_remote = (
        os.environ.get("VSCODE_REMOTE_AUTHORITY")
        or os.environ.get("SSH_CONNECTION")
        or os.environ.get("SSH_CLIENT")
    )

    if is_remote:
        matplotlib.use("module://ipympl.backend_nbagg")
        mne.viz.set_browser_backend("matplotlib")
    else:
        matplotlib.use("QtAgg")
        mne.viz.set_browser_backend("qt")