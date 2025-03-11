from __future__ import annotations

import os
from typing import Any

from utils import DRIVERS_TOOLS, ROOT, get_test_options, run_command


def set_env(name: str, value: Any = "1") -> None:
    os.environ[name] = str(value)


def start_server():
    opts, extra_opts = get_test_options(
        "Run a MongoDB server.  All given flags will be passed to run-orchestration.sh in DRIVERS_TOOLS.",
        require_sub_test_name=False,
        allow_extra_opts=True,
    )
    test_name = opts.test_name

    if opts.auth:
        extra_opts.append("--auth")

    if opts.verbose:
        extra_opts.append("-v")
    elif opts.quiet:
        extra_opts.append("-q")

    if test_name == "auth_aws":
        set_env("AUTH_AWS")

    elif test_name == "load_balancer":
        set_env("LOAD_BALANCER")

    elif test_name == "ocsp":
        opts.ssl = True
        if "ORCHESTRATION_FILE" not in os.environ:
            found = False
            for opt in extra_opts:
                if opt.startswith("--orchestration-file"):
                    found = True
            if not found:
                raise ValueError("Please provide an orchestration file")

    if not os.environ.get("TEST_CRYPT_SHARED"):
        set_env("SKIP_CRYPT_SHARED")

    if opts.ssl:
        extra_opts.append("--ssl")
        if test_name != "ocsp":
            certs = ROOT / "test/certificates"
            set_env("TLS_CERT_KEY_FILE", certs / "client.pem")
            set_env("TLS_PEM_KEY_FILE", certs / "server.pem")
            set_env("TLS_CA_FILE", certs / "ca.pem")

    cmd = ["bash", f"{DRIVERS_TOOLS}/.evergreen/run-orchestration.sh", *extra_opts]
    run_command(cmd, cwd=DRIVERS_TOOLS)


if __name__ == "__main__":
    start_server()
