"""python -m robot_session.manage --help"""

import argparse
import json
import sys
from pathlib import Path

from .backends import SessionHandle, build_image, replay, resume, start


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Explicit RoboCasa container lifecycle"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("backend", choices=("docker", "modal"))
    launch = commands.add_parser("start")
    launch.add_argument("backend", choices=("docker", "modal"))
    launch.add_argument("--image", required=True)
    launch.add_argument("--config", required=True)
    launch.add_argument("--receipt", required=True)
    launch.add_argument("--lifetime", type=int, default=86400)
    launch.add_argument("--cpu", type=float, default=4)
    launch.add_argument("--memory", type=int, default=16384)
    launch.add_argument("--startup-timeout", type=int, default=600)
    fresh = commands.add_parser("replay")
    fresh.add_argument("receipt")
    fresh.add_argument("--out", required=True)
    fresh.add_argument("--startup-timeout", type=int, default=600)
    restore = commands.add_parser("resume")
    restore.add_argument("checkpoint")
    restore.add_argument("--out", required=True)
    for name in ("status", "finish", "snapshot", "checkpoint", "run", "stop"):
        command = commands.add_parser(name)
        command.add_argument("receipt")
        if name in ("snapshot", "checkpoint"):
            command.add_argument("output")
        if name == "run":
            command.add_argument("source")
            command.add_argument("--timeout", type=int, default=600)
        if name == "stop":
            command.add_argument(
                "--snapshot", required=True, help="Export final state before teardown"
            )
    args = parser.parse_args(argv)
    if args.command == "build":
        print(build_image(args.backend))
    elif args.command == "start":
        handle = start(
            args.backend,
            args.image,
            json.loads(Path(args.config).read_text()),
            receipt=args.receipt,
            lifetime_seconds=args.lifetime,
            cpu=args.cpu,
            memory_mb=args.memory,
            startup_timeout=args.startup_timeout,
        )
        print(json.dumps(handle.status(), indent=2))
    elif args.command == "replay":
        handle = replay(
            SessionHandle.load(args.receipt),
            receipt=args.out,
            startup_timeout=args.startup_timeout,
        )
        print(json.dumps(handle.status(), indent=2))
    elif args.command == "resume":
        handle = resume(args.checkpoint, receipt=args.out)
        print(json.dumps(handle.status(), indent=2))
    else:
        handle = SessionHandle.load(args.receipt)
        if args.command == "run":
            result = handle.run_code(args.source, args.timeout)
            sys.stdout.buffer.write(result.stdout)
            sys.stderr.buffer.write(result.stderr)
            raise SystemExit(result.returncode)
        elif args.command == "snapshot":
            print(handle.snapshot(args.output))
        elif args.command == "checkpoint":
            record = handle.checkpoint(args.output)
            handle.save(args.receipt)
            print(json.dumps(record, indent=2))
        elif args.command == "stop":
            handle.finish()
            handle.snapshot(args.snapshot)
            handle.stop()
        else:
            print(json.dumps(getattr(handle, args.command)(), indent=2))


if __name__ == "__main__":
    main()
