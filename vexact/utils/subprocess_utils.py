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

import sys


def get_sys_executable():
    """
    Returns the appropriate command prefix for subprocess execution.

    If the current process is running under coverage (either via 'coverage run'
    or 'pytest --cov'), returns a command that will run subprocesses under
    coverage with --append flag. Otherwise, returns the standard Python executable.

    Returns:
        list: Command prefix to use for subprocess execution
              e.g., [sys.executable] or [sys.executable, "-m", "coverage", "run", "--parallel-mode"]
    """

    if "coverage" in sys.modules:
        try:
            import coverage

            if coverage.Coverage.current() is not None:
                return [sys.executable, "-m", "coverage", "run", "--parallel-mode"]
        except (AttributeError, ImportError):
            pass

    return [sys.executable]
