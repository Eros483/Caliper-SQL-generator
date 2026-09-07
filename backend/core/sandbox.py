import os
import shutil
import tempfile

import docker
from docker.errors import ImageNotFound

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

IMAGE_NAME = "caliperlens-sandbox"
IMAGE_TAG = "latest"


class SandboxError(Exception):
    pass


class SandboxTimeout(SandboxError):
    pass


class SandboxExecutor:
    def __init__(self, image_name: str = IMAGE_NAME, db_path: str | None = None):
        self.image_name = image_name
        self.db_path = db_path or settings.duckdb_path
        self._docker = None
        self._image_built = False

    @property
    def docker_client(self):
        if self._docker is None:
            try:
                self._docker = docker.from_env()
                self._docker.ping()
            except Exception as e:
                raise SandboxError(f"Docker not available: {e}")
        return self._docker

    def ensure_image(self) -> None:
        if self._image_built:
            return
        try:
            self.docker_client.images.get(f"{self.image_name}:{IMAGE_TAG}")
            self._image_built = True
            logger.info(f"Sandbox image '{self.image_name}' found.")
        except ImageNotFound:
            logger.info(f"Building sandbox image '{self.image_name}'...")
            sandbox_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sandbox")
            _, logs = self.docker_client.images.build(
                path=sandbox_dir,
                tag=f"{self.image_name}:{IMAGE_TAG}",
                rm=True,
            )
            for line in logs:
                if "stream" in line:
                    logger.debug(line["stream"].strip())
            self._image_built = True
            logger.info(f"Sandbox image '{self.image_name}' built.")

    def execute(self, code: str, timeout: int = 30) -> dict[str, object]:
        self.ensure_image()

        work_dir = tempfile.mkdtemp(prefix="sandbox_")
        script_path = os.path.join(work_dir, "script.py")

        try:
            with open(script_path, "w") as f:
                f.write(code)

            db_mount_path = os.path.abspath(self.db_path)
            db_dir = os.path.dirname(db_mount_path)

            volumes = {
                work_dir: {"bind": "/code", "mode": "rw"},
            }

            if os.path.exists(db_mount_path):
                volumes[db_dir] = {"bind": "/data", "mode": "ro"}

            container = self.docker_client.containers.run(
                f"{self.image_name}:{IMAGE_TAG}",
                command=["/entrypoint.sh"],
                environment={"SCRIPT_FILE": "/code/script.py"},
                volumes=volumes,
                network_mode="none",
                mem_limit="256m",
                nano_cpus=1_000_000_000,
                pids_limit=50,
                detach=True,
                remove=False,
            )

            try:
                result = container.wait(timeout=timeout)
                exit_code = result.get("StatusCode", -1)
            except Exception:
                container.kill()
                container.remove(force=True)
                raise SandboxTimeout(f"Sandbox execution exceeded {timeout}s timeout")

            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

            container.remove()

            artifacts = []
            for fname in os.listdir(work_dir):
                fpath = os.path.join(work_dir, fname)
                if fname != "script.py" and os.path.isfile(fpath):
                    with open(fpath, "rb") as af:
                        artifacts.append(
                            {
                                "filename": fname,
                                "data": af.read(),
                            }
                        )

            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "artifacts": artifacts,
            }

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def health_check(self) -> bool:
        try:
            self.ensure_image()
            result = self.execute("print('sandbox_ok')", timeout=10)
            return result["exit_code"] == 0 and "sandbox_ok" in str(result["stdout"])
        except Exception as e:
            logger.error(f"Sandbox health check failed: {e}")
            return False


# ponytail: minimal seam — same execute/health_check contract for both backends
class _LambdaSandboxStub:
    def execute(self, code: str, timeout: int = 30) -> dict[str, object]:
        raise NotImplementedError("Lambda sandbox not configured in local env")

    def health_check(self) -> bool:
        return False


def get_sandbox_executor() -> SandboxExecutor | _LambdaSandboxStub:
    backend = (settings.sandbox_backend or "local").lower()
    if backend == "lambda":
        return _LambdaSandboxStub()
    return SandboxExecutor()
