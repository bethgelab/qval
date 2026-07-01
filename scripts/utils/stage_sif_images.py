#!/usr/bin/env python3
"""Stage SIF images for TerminalBench.

Run this on a login node (with apptainer/singularity and network access)
to pre-build SIF images into a shared cache directory. GPU compute nodes
will then consume these cached images at runtime.

Usage:
    python scripts/stage_sif_images.py \
        --env-name "terminal-bench@2.0" \
        --sif-cache-dir /shared/sif_cache \
        --apptainer-command apptainer

    # With preinstall overrides:
    python scripts/stage_sif_images.py \
        --env-name "terminal-bench@2.0" \
        --sif-cache-dir /shared/sif_cache \
        --preinstall-config configs/image_preinstalls/terminal_bench.yaml

    # Dry-run (show what would be built, don't build):
    python scripts/stage_sif_images.py \
        --env-name "terminal-bench@2.0" \
        --sif-cache-dir /shared/sif_cache \
        --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _sif_cache_key(image_ref: str) -> str:
    return hashlib.sha256(image_ref.encode()).hexdigest()[:16]


def _preinstall_hash(post_install: list[str]) -> str:
    content = "\n".join(post_install)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PreinstallConfig:
    """Curated preinstall rules for SIF staging."""

    global_post_install: tuple[str, ...] = ()
    image_overrides: dict[str, tuple[str, ...]] = field(default_factory=dict)


def load_preinstall_config(path: Path) -> PreinstallConfig:
    """Load curated preinstall rules from YAML.

    Entries with invalid format are skipped with a warning.
    """
    if not path.exists():
        return PreinstallConfig()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        return PreinstallConfig()
    if "all" in data and "*" in data:
        raise ValueError("Preinstall config must not define both 'all' and '*'")

    global_post_install: tuple[str, ...] = ()
    result: dict[str, tuple[str, ...]] = {}
    for ref, entry in data.items():
        if not isinstance(ref, str):
            continue
        if not isinstance(entry, dict) or "post_install" not in entry:
            continue
        post_install = entry["post_install"]
        if not isinstance(post_install, list) or not all(
            isinstance(cmd, str) for cmd in post_install
        ):
            print(
                f"  WARNING: skipping {ref}: post_install must be a list of strings",
                file=sys.stderr,
            )
            continue
        normalized = tuple(post_install)
        if ref in {"all", "*"}:
            global_post_install = normalized
        else:
            result[ref] = normalized
    return PreinstallConfig(
        global_post_install=global_post_install,
        image_overrides=result,
    )


def resolve_preinstall_commands(
    config: PreinstallConfig,
    *,
    image_ref: str,
    source: str,
) -> list[str] | None:
    """Resolve merged preinstall commands for an image."""
    commands = list(config.global_post_install)
    specific = config.image_overrides.get(image_ref) or config.image_overrides.get(
        source
    )
    if specific:
        commands.extend(specific)
    return commands or None


def _load_preinstall_hashes(cache_dir: Path) -> dict[str, str]:
    path = cache_dir / "preinstall_hashes.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_preinstall_hashes(cache_dir: Path, hashes: dict[str, str]) -> None:
    path = cache_dir / "preinstall_hashes.json"
    path.write_text(json.dumps(hashes, indent=2, sort_keys=True))


def _needs_rebuild(
    image_ref: str,
    sif_path: Path,
    post_install: list[str] | None,
    preinstall_hashes: dict[str, str],
    force: bool,
) -> bool:
    if force:
        return True
    if not sif_path.exists():
        return True
    if not post_install:
        return False
    current_hash = _preinstall_hash(post_install)
    return preinstall_hashes.get(image_ref) != current_hash


@dataclass
class _DockerfileParseResult:
    """Parsed Dockerfile content ready for Singularity .def generation."""

    base_image: str
    files: list[tuple[str, str]]  # (host_src, container_dst)
    post_commands: list[str]
    environment: list[tuple[str, str]]  # (key, value)
    startscript: str | None = None
    # Multi-stage: if set, this stage must be built first.
    builder_stage: "_DockerfileParseResult | None" = None
    builder_name: str | None = None


def _parse_dockerfile(dockerfile_path: Path) -> _DockerfileParseResult:
    """Parse a Dockerfile into components for Singularity .def generation.

    Handles single-stage and simple two-stage (builder pattern) Dockerfiles.
    """
    context_dir = dockerfile_path.parent
    lines = dockerfile_path.read_text().splitlines()

    # Join continuation lines (trailing backslash).
    joined: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        if joined and joined[-1].endswith("\\"):
            joined[-1] = joined[-1][:-1] + stripped
        else:
            joined.append(stripped)

    # Split into stages on FROM instructions.
    stages: list[tuple[str, str | None, list[str]]] = []  # (image, alias, lines)
    for line in joined:
        bare = line.lstrip()
        if bare.upper().startswith("FROM "):
            parts = bare.split()
            image = parts[1]
            alias = None
            if len(parts) >= 4 and parts[2].upper() == "AS":
                alias = parts[3]
            stages.append((image, alias, []))
        elif stages:
            stages[-1][2].append(line)

    if not stages:
        raise ValueError(f"No FROM instruction found in {dockerfile_path}")

    def _parse_stage(
        image: str,
        body: list[str],
        ctx: Path,
        *,
        builder_sif_dir: Path | None = None,
        builder_name: str | None = None,
    ) -> _DockerfileParseResult:
        files: list[tuple[str, str]] = []
        post: list[str] = []
        env: list[tuple[str, str]] = []
        startscript: str | None = None
        workdir = ""

        for line in body:
            bare = line.lstrip()
            if not bare or bare.startswith("#"):
                continue
            upper = bare.split()[0].upper() if bare.split() else ""

            if upper == "RUN":
                cmd = bare[3:].lstrip()
                if workdir:
                    cmd = f"cd {workdir} && {cmd}"
                post.append(cmd)

            elif upper == "COPY":
                rest = bare[4:].lstrip()
                if rest.startswith("--from="):
                    # Multi-stage COPY: source is from builder rootfs.
                    # Store with a marker prefix; resolved at build time.
                    flag, *parts = rest.split()
                    if len(parts) >= 2:
                        src, dst = parts[0], parts[-1]
                        files.append((f"__from_builder__:{src}", dst))
                    continue
                parts = rest.rsplit(None, 1)
                if len(parts) == 2:
                    src_glob, dst = parts
                    # Source is relative to the Dockerfile's directory.
                    for src_token in src_glob.split():
                        host_src = str(ctx / src_token)
                        files.append((host_src, dst))

            elif upper == "WORKDIR":
                workdir = bare[7:].lstrip()
                post.append(f"mkdir -p {workdir}")

            elif upper == "ENV":
                rest = bare[3:].lstrip()
                if "=" in rest:
                    key, _, val = rest.partition("=")
                    env.append((key.strip(), val.strip()))
                    post.append(f"export {key.strip()}={val.strip()}")
                else:
                    parts = rest.split(None, 1)
                    if len(parts) == 2:
                        env.append((parts[0], parts[1]))
                        post.append(f"export {parts[0]}={parts[1]}")

            elif upper == "USER":
                user = bare[4:].lstrip()
                # Singularity %post always runs as root; track user switches
                # as su commands for subsequent RUN lines if needed.
                # For simplicity, just note it — most tasks run as root.
                pass

            elif upper in ("CMD", "ENTRYPOINT"):
                rest = bare[len(upper) :].lstrip()
                if rest.startswith("["):
                    # JSON array form: ["cmd", "arg1", ...]
                    import ast

                    try:
                        parts = ast.literal_eval(rest)
                        if isinstance(parts, list) and parts:
                            startscript = " ".join(parts)
                    except (ValueError, SyntaxError):
                        pass
                elif rest:
                    startscript = rest

            # EXPOSE, ARG, LABEL etc. — ignored for SIF builds.

        return _DockerfileParseResult(
            base_image=image,
            files=files,
            post_commands=post,
            environment=env,
            startscript=startscript,
        )

    if len(stages) == 1:
        image, _, body = stages[0]
        return _parse_stage(image, body, context_dir)

    # Two-stage build: build the first stage, then the final stage.
    builder_image, builder_alias, builder_body = stages[0]
    final_image, _, final_body = stages[-1]

    builder = _parse_stage(builder_image, builder_body, context_dir)
    builder.builder_name = builder_alias

    result = _parse_stage(
        final_image,
        final_body,
        context_dir,
        builder_name=builder_alias,
    )
    result.builder_stage = builder
    return result


def _generate_def_file(
    parsed: _DockerfileParseResult,
    *,
    post_install: list[str] | None = None,
) -> str:
    """Generate a Singularity .def file from parsed Dockerfile content."""
    sections: list[str] = []

    # Header
    sections.append(f"Bootstrap: docker\nFrom: {parsed.base_image}\n")

    # %files — COPY instructions (skip unresolved __from_builder__ markers)
    all_files = [
        (s, d) for s, d in parsed.files if not s.startswith("__from_builder__:")
    ]
    if all_files:
        file_lines = ["%files"]
        for src, dst in all_files:
            file_lines.append(f"    {src} {dst}")
        sections.append("\n".join(file_lines) + "\n")

    # %environment
    if parsed.environment:
        env_lines = ["%environment"]
        for key, val in parsed.environment:
            env_lines.append(f"    export {key}={val}")
        sections.append("\n".join(env_lines) + "\n")

    # %post — RUN instructions + preinstall
    post_lines = list(parsed.post_commands)
    if post_install:
        post_lines.extend(post_install)
    if post_lines:
        sections.append("%post\n" + "\n".join(post_lines) + "\n")

    # %startscript — CMD/ENTRYPOINT
    if parsed.startscript:
        sections.append(f"%startscript\n{parsed.startscript}\n")

    return "\n".join(sections)


def build_from_dockerfile(
    apptainer: str,
    dockerfile_path: Path,
    target_sif: Path,
    *,
    post_install: list[str] | None = None,
    fakeroot: bool = False,
) -> None:
    """Build SIF from a Dockerfile by converting to a Singularity .def file."""
    parsed = _parse_dockerfile(dockerfile_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Handle multi-stage: build builder stage first, extract its rootfs.
        if parsed.builder_stage:
            builder_def = _generate_def_file(parsed.builder_stage)
            builder_def_path = tmp / "builder.def"
            builder_def_path.write_text(builder_def)
            builder_sif = tmp / "builder.sif"

            cmd = [apptainer, "build", "--force"]
            if fakeroot:
                cmd.append("--fakeroot")
            cmd.extend([str(builder_sif), str(builder_def_path)])
            subprocess.run(cmd, check=True)

            # Extract builder rootfs so COPY --from= can find files.
            builder_rootfs = tmp / "builder_rootfs"
            cmd = [apptainer, "build", "--force", "--sandbox"]
            if fakeroot:
                cmd.append("--fakeroot")
            cmd.extend([str(builder_rootfs), str(builder_sif)])
            subprocess.run(cmd, check=True)

            # Resolve __from_builder__: markers to real host paths.
            fixed_files = []
            for src, dst in parsed.files:
                if src.startswith("__from_builder__:"):
                    container_path = src.split(":", 1)[1]
                    fixed_files.append(
                        (str(builder_rootfs / container_path.lstrip("/")), dst)
                    )
                else:
                    fixed_files.append((src, dst))
            parsed.files = fixed_files

        def_content = _generate_def_file(parsed, post_install=post_install)
        def_path = tmp / "converted.def"
        def_path.write_text(def_content)

        cmd = [apptainer, "build", "--force"]
        if fakeroot:
            cmd.append("--fakeroot")
        cmd.extend([str(target_sif), str(def_path)])
        subprocess.run(cmd, check=True)


def build_with_preinstall(
    apptainer: str,
    source: str,
    target_sif: Path,
    post_install: list[str],
    *,
    fakeroot: bool = False,
) -> None:
    """Build SIF with post-install commands via two-stage localimage bootstrap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        base_sif = tmp / "base.sif"

        # Stage 1: build base SIF from original source
        cmd = [apptainer, "build", "--force"]
        if fakeroot:
            cmd.append("--fakeroot")
        cmd.extend([str(base_sif), source])
        subprocess.run(cmd, check=True)

        # Stage 2: build augmented SIF from localimage + %post
        def_file = tmp / "augmented.def"
        post_block = "\n".join(post_install)
        def_file.write_text(
            f"Bootstrap: localimage\nFrom: {base_sif}\n\n%post\n{post_block}\n"
        )
        cmd = [apptainer, "build", "--force"]
        if fakeroot:
            cmd.append("--fakeroot")
        cmd.extend([str(target_sif), str(def_file)])
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage SIF images for Apptainer-HPC")
    parser.add_argument(
        "--env-name", default="terminal-bench@2.0", help="Harbor env name"
    )
    parser.add_argument(
        "--dataset-path", default=None, help="Local Harbor dataset path"
    )
    parser.add_argument(
        "--sif-cache-dir", required=True, help="Output directory for SIF files"
    )
    parser.add_argument(
        "--apptainer-command", default="apptainer", help="apptainer or singularity"
    )
    parser.add_argument(
        "--preinstall-config",
        default=None,
        help="Path to YAML preinstall overrides (e.g., configs/image_preinstalls/terminal_bench.yaml)",
    )
    parser.add_argument(
        "--docker-archive-dir",
        default=None,
        help="Directory of pre-built Docker tars (<task_name>.tar). "
        "Used for Dockerfile-only tasks instead of building from the Dockerfile directly.",
    )
    parser.add_argument(
        "--fakeroot", action="store_true", help="Use --fakeroot for builds"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be built"
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if SIF exists"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of image builds to run in parallel. Each build uses ~3-5GB "
        "RAM and saturates ~1 core; the practical ceiling is shared network "
        "bandwidth to package mirrors (typical sweet spot: 4-16).",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    cache_dir = Path(args.sif_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest_path = cache_dir / "manifest.json"
    manifest: dict[str, str] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    # Load preinstall config
    preinstall_overrides = PreinstallConfig()
    if args.preinstall_config:
        preinstall_overrides = load_preinstall_config(Path(args.preinstall_config))
        if (
            preinstall_overrides.global_post_install
            or preinstall_overrides.image_overrides
        ):
            print(
                "Loaded preinstall rules:"
                f" global={'yes' if preinstall_overrides.global_post_install else 'no'},"
                f" image-specific={len(preinstall_overrides.image_overrides)}"
            )

    preinstall_hashes = _load_preinstall_hashes(cache_dir)

    # Load tasks
    from llenvs.adapters.harbor import HarborAdapter

    adapter = HarborAdapter()
    tasks = adapter.load_tasks(args.env_name, dataset_path=args.dataset_path)
    print(f"Loaded {len(tasks)} tasks from {args.env_name}")

    # Analyze each task
    to_build: list[tuple[str, str, Path, list[str] | None]] = []
    skipped_compose = 0
    skipped_no_source = 0
    already_cached = 0

    for task in tasks:
        env_dir = Path(task.paths.environment_dir)
        compose_path = env_dir / "docker-compose.yaml"
        dockerfile_path = env_dir / "Dockerfile"
        docker_image = getattr(task.config.environment, "docker_image", None)

        if compose_path.exists():
            skipped_compose += 1
            continue

        if docker_image:
            image_ref = docker_image
            source = f"docker://{docker_image}"
        elif dockerfile_path.exists():
            image_ref = f"dockerfile://{task.name}"
            # Prefer pre-built docker archive if available.
            if args.docker_archive_dir:
                tar_path = Path(args.docker_archive_dir) / f"{task.name}.tar"
                if tar_path.exists():
                    source = f"docker-archive:{tar_path}"
                else:
                    print(
                        f"  WARNING: no archive for {task.name}, falling back to Dockerfile",
                        file=sys.stderr,
                    )
                    source = str(dockerfile_path)
            else:
                source = str(dockerfile_path)
        else:
            skipped_no_source += 1
            continue

        # SIF filename always uses the base image_ref hash (runtime expects this)
        cache_key = _sif_cache_key(image_ref)
        sif_path = cache_dir / f"{cache_key}.sif"

        post_install = resolve_preinstall_commands(
            preinstall_overrides,
            image_ref=image_ref,
            source=source,
        )

        if not _needs_rebuild(
            image_ref, sif_path, post_install, preinstall_hashes, args.force
        ):
            already_cached += 1
            manifest[image_ref] = sif_path.name
            continue

        to_build.append((image_ref, task.name, source, sif_path, post_install))

    print("\nSummary:")
    print(
        f"  Single-container tasks: {len(tasks) - skipped_compose - skipped_no_source}"
    )
    print(f"  Compose tasks (skipped): {skipped_compose}")
    print(f"  No container source (skipped): {skipped_no_source}")
    print(f"  Already cached: {already_cached}")
    print(f"  To build: {len(to_build)}")

    if args.dry_run:
        for image_ref, task_name, source, sif_path, post_install in to_build:
            print(f"\n  Would build: {sif_path.name}")
            print(f"    Task: {task_name}")
            print(f"    Source: {source}")
            if post_install:
                print(f"    Preinstall: {len(post_install)} command(s)")
        return

    # Build SIF images
    def _build_one(
        image_ref: str,
        task_name: str,
        source: str,
        sif_path: Path,
        post_install: list[str] | None,
    ) -> tuple[str, str, Path, list[str] | None, BaseException | None]:
        """Run a single build; return exception (if any) for main thread."""
        try:
            is_raw_dockerfile = image_ref.startswith(
                "dockerfile://"
            ) and not source.startswith("docker-archive:")
            if is_raw_dockerfile:
                build_from_dockerfile(
                    args.apptainer_command,
                    Path(source),
                    sif_path,
                    post_install=post_install,
                    fakeroot=args.fakeroot,
                )
            elif post_install:
                build_with_preinstall(
                    args.apptainer_command,
                    source,
                    sif_path,
                    post_install,
                    fakeroot=args.fakeroot,
                )
            else:
                cmd = [args.apptainer_command, "build", "--force"]
                if args.fakeroot:
                    cmd.append("--fakeroot")
                cmd.extend([str(sif_path), source])
                subprocess.run(cmd, check=True)
            return (image_ref, task_name, sif_path, post_install, None)
        except BaseException as exc:  # noqa: BLE001 — surface every failure to caller
            return (image_ref, task_name, sif_path, post_install, exc)

    manifest_lock = threading.Lock()
    print_lock = threading.Lock()
    failed = 0
    completed = 0
    total = len(to_build)

    print(f"\nBuilding {total} images with --jobs {args.jobs} ...")

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                _build_one, image_ref, task_name, source, sif_path, post_install
            ): (image_ref, task_name, source, sif_path, post_install)
            for image_ref, task_name, source, sif_path, post_install in to_build
        }

        # Announce dispatch (kept simple because order is now non-deterministic).
        for image_ref, task_name, source, sif_path, post_install in to_build:
            if post_install:
                desc = (
                    f"{sif_path.name} from {source} + {len(post_install)} preinstall(s)"
                )
            else:
                desc = f"{sif_path.name} from {source}"
            with print_lock:
                print(f"  QUEUED: {task_name} ({desc})")

        for future in as_completed(futures):
            image_ref, task_name, sif_path, post_install, exc = future.result()
            completed += 1
            label = f"[{completed}/{total}]"
            if exc is None:
                with manifest_lock:
                    manifest[image_ref] = sif_path.name
                    if post_install:
                        preinstall_hashes[image_ref] = _preinstall_hash(post_install)
                    elif image_ref in preinstall_hashes:
                        del preinstall_hashes[image_ref]
                with print_lock:
                    print(f"{label} OK: {task_name} -> {sif_path.name}")
            else:
                failed += 1
                with print_lock:
                    print(
                        f"{label} FAILED: {task_name} -> {sif_path.name}: {exc}",
                        file=sys.stderr,
                    )

    # Save manifest and preinstall hashes (single flush, thread-safe because
    # all workers have joined).
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nManifest written to {manifest_path}")
    _save_preinstall_hashes(cache_dir, preinstall_hashes)

    if failed:
        print(f"\n{failed} builds failed!", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. {total - failed} images built, {already_cached} cached.")


if __name__ == "__main__":
    main()
