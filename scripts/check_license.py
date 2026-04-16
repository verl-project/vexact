# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check and optionally add Apache 2.0 license headers to Python files."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable


LICENSE_HEADERS = [
    "Copyright 2024 Bytedance Ltd. and/or its affiliates",
    "Copyright 2025 Bytedance Ltd. and/or its affiliates",
    "Copyright 2026 Bytedance Ltd. and/or its affiliates",
    "Copyright 2025 Individual Contributor:",
    # Third-party files adapted under their original licenses
    "Modifications copyright (c) 2025 ByteDance Ltd. and/or its affiliates",
    "Modifications copyright (c) 2026 ByteDance Ltd. and/or its affiliates",
    "Copyright (c) [year] sail-sg/Precision-RL-verl",
]

LICENSE_TEMPLATE = """\
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""


def get_py_files(path_arg: Path) -> Iterable[Path]:
    if path_arg.is_dir():
        return path_arg.glob("**/*.py")
    elif path_arg.is_file() and path_arg.suffix == ".py":
        return [path_arg]
    return []


def has_license(content: str) -> bool:
    return any(header in content for header in LICENSE_HEADERS)


def add_license(path: Path, content: str) -> None:
    # Preserve shebang or encoding declarations at the top
    lines = content.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#!") or stripped.startswith("# -*-") or stripped.startswith("# coding"):
            insert_at = i + 1
        else:
            break
    new_content = "".join(lines[:insert_at]) + LICENSE_TEMPLATE + "\n" + "".join(lines[insert_at:])
    path.write_text(new_content, encoding="utf-8")
    print(f"  [added]   {path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Check and optionally add license headers to Python files.")
    parser.add_argument(
        "--directories",
        "-d",
        required=True,
        type=Path,
        nargs="+",
        help="Directories (or files) to check",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        default=False,
        help="Automatically add the license header to files that are missing it",
    )
    args = parser.parse_args()

    pathlist = sorted(set(path for path_arg in args.directories for path in get_py_files(path_arg)))

    missing = []
    for path in pathlist:
        content = path.read_text(encoding="utf-8")
        if not has_license(content):
            missing.append(path)
            if args.add:
                add_license(path, content)
            else:
                print(f"  [missing] {path}")

    if missing:
        if args.add:
            print(f"\nAdded license header to {len(missing)} file(s).")
        else:
            print(f"\n{len(missing)} file(s) are missing a license header. Re-run with --add to fix.")
            raise SystemExit(1)
    else:
        print("All files have a license header.")
