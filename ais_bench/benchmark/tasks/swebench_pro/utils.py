import os
import subprocess
from typing import Callable, Iterable, TypeVar

from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import SWEB_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchImportError, AISBenchRuntimeError

DATASET_MAPPING = {
    "full": "ScaleAI/SWE-bench_Pro",
    "mini": "ScaleAI/SWE-bench_Pro_Mini",
}


def cleanup_swebench_pro_containers():
    name_filters = ["minisweagent-", "sweb.eval"]
    for name_filter in name_filters:
        try:
            r = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"name={name_filter}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode != 0 or not (r.stdout or "").strip():
                continue
            ids = [x.strip() for x in r.stdout.strip().splitlines() if x.strip()]
            if not ids:
                continue
            subprocess.run(
                ["docker", "rm", "-f"] + ids,
                capture_output=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass


def list_swebench_pro_images(client) -> set[str]:
    """列出当前所有 SWE-bench Pro 镜像（jefzda/sweap-images:*）"""
    existing_images = set()
    try:
        for image in client.images.list(all=True):
            for tag in image.tags:
                if tag.startswith("jefzda/sweap-images:"):
                    existing_images.add(tag)
    except Exception:
        pass
    return existing_images


def remove_swebench_pro_image(client, image_tag: str, logger: AISLogger):
    """删除单个 SWE-bench Pro 镜像"""
    try:
        client.images.remove(image_tag, force=True)
        logger.debug(f"Removed image: {image_tag}")
    except Exception as e:
        logger.warning(f"Failed to remove image {image_tag}: {e}")


def clean_swebench_pro_images(client, prior_images: set[str], logger: AISLogger):
    """清理评测过程中拉取的新镜像（不在原有列表中的）"""
    current_images = list_swebench_pro_images(client)
    new_images = current_images - prior_images
    
    if new_images:
        logger.info("Cleaning up %d new SWE-bench Pro images...", len(new_images))
        for image_tag in new_images:
            remove_swebench_pro_image(client, image_tag, logger)
        logger.info("Image cleanup completed.")
    else:
        logger.debug("No new images to clean up.")


def docker_image_exists_locally(image: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return r.returncode == 0
    except Exception:
        return False


def docker_pull_image(image: str, logger: AISLogger) -> bool:
    logger.info("Pulling Docker image: %s", image)
    r = subprocess.run(["docker", "pull", image])
    return r.returncode == 0


_T = TypeVar("_T")


def ensure_swebench_pro_docker_images(
    items: Iterable[_T],
    logger: AISLogger,
    get_image_name: Callable[[_T], str],
    *,
    task_label: str = "infer",
) -> None:
    ordered_unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = get_image_name(item)
        if name not in seen:
            seen.add(name)
            ordered_unique.append(name)

    failed: list[str] = []
    for image in ordered_unique:
        if docker_image_exists_locally(image):
            logger.debug("Docker image already present: %s", image)
            continue
        if docker_pull_image(image, logger):
            if docker_image_exists_locally(image):
                continue
        failed.append(image)

    if failed:
        raise AISBenchRuntimeError(
            SWEB_CODES.DOCKER_IMAGE_UNAVAILABLE,
            "Required SWE-bench Pro Docker image(s) missing or pull failed; "
            f"aborting {task_label}. Images: {failed}"
        )


def build_problem_statement(row: dict) -> str:
    parts = [row["problem_statement"]]
    if row.get("requirements"):
        parts.append(f"\nRequirements:\n{row['requirements']}")
    if row.get("interface"):
        parts.append(f"\nNew interfaces introduced::\n{row['interface']}")
    return "\n".join(parts)


def get_dockerhub_image_uri(raw_instance: dict) -> str:
    uid = raw_instance["instance_id"]
    repo_name = raw_instance["repo"]
    repo_base, repo_name_only = repo_name.lower().split("/")
    hsh = uid.replace("instance_", "")

    if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = 'element-web'  # Keep full name for this one case
    elif 'element-hq' in repo_name.lower() and 'element-web' in repo_name.lower():
        repo_name_only = 'element'
        if hsh.endswith('-vnan'):
            hsh = hsh[:-5]
    # All other repos: strip -vnan suffix
    elif hsh.endswith('-vnan'):
        hsh = hsh[:-5]
    
    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]
    
    return f"jefzda/sweap-images:{tag}"


def build_instance(raw_instance: dict) -> dict:
    return {
        "instance_id": raw_instance["instance_id"],
        "problem_statement": build_problem_statement(raw_instance),
        "repo_name": raw_instance["repo"],
        "base_commit": raw_instance["base_commit"],
        "image_name": get_dockerhub_image_uri(raw_instance),
    }


def eval_with_docker(patch, sample, output_dir, scripts_dir, prefix="", docker_client=None, timeout=7200):
    if docker_client is None:
        try:
            import docker
        except Exception:
            raise RuntimeError("docker SDK is not installed. Install via 'pip install docker' or run without --use_local_docker")
        docker_client = docker.from_env()
 
    try:
        from swe_bench_pro_eval import (
            prepare_run,
            assemble_workspace_files,
            write_files_local,
            write_patch_snapshot,
            collect_outputs_local,
            save_entryscript_copy
        )
    except ImportError as e:
        raise AISBenchImportError(
            SWEB_CODES.SWEBENCH_HARNESS_IMPORT_ERROR,
            "SWEBenchEvalTask requires the SWE-bench harness. "
            "Install from: https://github.com/SWE-bench/SWE-bench"
        ) from e

    uid = sample["instance_id"]
    existing_output, output_path, workspace_dir = prepare_run(uid, output_dir, prefix, False)
    if existing_output is not None:
        return existing_output

    print(f"Running local-docker evaluation for {uid}")

    try:
        try:
            files, entryscript_content = assemble_workspace_files(uid, scripts_dir, patch, sample)
        except FileNotFoundError as e:
            print(f"Error loading scripts for {uid}: {e}")
            return None
        write_files_local(workspace_dir, files)
        write_patch_snapshot(output_dir, uid, prefix, patch)

        abs_workspace_dir = os.path.abspath(workspace_dir)
        volumes = {abs_workspace_dir: {"bind": "/workspace", "mode": "rw"}}
        run_kwargs = {
            "volumes": volumes,
            "detach": True,
            "remove": True,
            "entrypoint": "/bin/bash",  # Override image entrypoint
            "command": ["-c", "bash /workspace/entryscript.sh"],
        }

        dockerhub_image_uri = get_dockerhub_image_uri(sample)
        print(f"Using image: {dockerhub_image_uri}")
        container = docker_client.containers.run(
            dockerhub_image_uri,
            **run_kwargs,
        )

        try:
            result = container.wait(timeout=timeout)
            status_code = result.get("StatusCode", 1) if isinstance(result, dict) else 1
        except docker.errors.Timeout:
            print(f"Container for {uid} timed out after {timeout}s, terminating...")
            try:
                container.stop(timeout=10)  # 10秒优雅停止
            except Exception:
                container.kill()  # 强制杀死
            return {
                "tests": [],
                "error": "timeout",
                "message": f"Evaluation timed out after {timeout} seconds",
                "instance_id": uid,
            }

        if status_code != 0:
            print(f"Entryscript failed for {uid} with return code: {status_code}")
        output = collect_outputs_local(workspace_dir, output_dir, uid, prefix)
        if output is None:
            return None
        save_entryscript_copy(output_dir, uid, prefix, entryscript_content)

        return output
    except Exception as e:
        print(f"Error in eval_with_docker for {uid}: {repr(e)}")
        print(f"Error type: {type(e)}")
        return None
