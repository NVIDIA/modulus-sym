# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import os
import time
import random
from termcolor import colored
from collections import Counter
import itertools
from process import Process


class Que:
    def __init__(
        self,
        available_gpus=(0, 1),
        print_type="que",
        exit_on_fail=True,
        loop_pause=10.0,
        clear_prints=False,
    ):
        self.pl = []
        self.pl_errors = []  # Holds failed processes
        self.running_pl = []
        self.available_gpus = list(available_gpus)
        self.start_time = 0
        self.print_type = print_type
        self.exit_on_fail = exit_on_fail
        self.loop_pause = loop_pause
        self.clear_prints = clear_prints

    def enque_experiments(self, cmd, params, cwd="./"):
        cross_params = itertools.product(
            *[
                ["--" + key + "=" + str(value) for value in params[key]]
                for key in params.keys()
            ]
        )
        cmds = []
        cwds = []
        for param in cross_params:
            cmds.append(cmd + " " + " ".join(param))
            cwds.append(cwd)
        self.enque_cmds(cmds, cwds)

    def enque_cmds(self, name, cmds, cwds):
        if type(cmds) is str:
            cmds = [cmds]
            cwds = [cwds]
        random.shuffle(cmds)
        for cmd, cwd in zip(cmds, cwds):
            self.pl.append(Process(name, cmd, cwd=cwd))

    def start_next(self, gpu):
        for i in range(len(self.pl)):
            if self.pl[i].get_status() == "Not Started":
                print(colored(f"Starting job: {self.pl[i].name}", "yellow"))
                self.pl[i].start(gpu)
                break

    def find_free_gpu(self):
        used_gpus = []
        for i in range(len(self.pl)):
            if self.pl[i].get_status() == "Running":
                used_gpus.append(self.pl[i].get_gpu())
        free_gpus = list(Counter(self.available_gpus) - Counter(used_gpus))
        return free_gpus

    def num_finished_processes(self):
        proc = 0
        for i in range(len(self.pl)):
            if (
                self.pl[i].get_status() == "Finished"
                and self.pl[i].get_return_status() == "SUCCESS"
            ):
                proc += 1
        return proc

    def num_failed_processes(self):
        proc = 0
        for i in range(len(self.pl)):
            if (
                self.pl[i].get_status() == "Finished"
                and self.pl[i].get_return_status() == "FAIL"
            ):
                proc += 1
        return proc

    def num_running_processes(self):
        proc = 0
        for i in range(len(self.pl)):
            if self.pl[i].get_status() == "Running":
                proc += 1
        return proc

    def num_unstarted_processes(self):
        proc = 0
        for i in range(len(self.pl)):
            if self.pl[i].get_status() == "Not Started":
                proc += 1
        return proc

    def percent_complete(self):
        rc = 0.0
        if self.num_finished_processes() > 0:
            rc = self.num_finished_processes() / float(len(self.pl))
        return rc * 100.0

    def run_time(self):
        return time.time() - self.start_time

    def time_left(self):
        tl = -1
        pc = self.percent_complete()
        if pc > 0:
            tl = (time.time() - self.start_time) * (
                1.0 / (pc / 100.0)
            ) - self.run_time()
        return tl

    def time_string(self, tl):
        tl = max([0, tl])
        seconds = int(tl % 60)
        tl = (tl - seconds) / 60
        mins = int(tl % 60)
        tl = (tl - mins) / 60
        hours = int(tl % 24)
        days = int((tl - hours) / 24)
        return (
            "("
            + str(days).zfill(3)
            + ":"
            + str(hours).zfill(2)
            + ":"
            + str(mins).zfill(2)
            + ":"
            + str(seconds).zfill(2)
            + ")"
        )

    def update_pl_status(self):
        for i in range(len(self.pl)):
            self.pl[i].update_status()

    def print_que_status(self):
        if self.clear_prints:
            os.system("clear")
        print("QUE STATUS")
        print(
            colored(
                "Num Finished Success: " + str(self.num_finished_processes()), "green"
            )
        )
        print(
            colored("Num Finished Fail:    " + str(self.num_failed_processes()), "red")
        )
        print(
            colored(
                "Num Running:          " + str(self.num_running_processes()), "yellow"
            )
        )
        print(
            colored(
                "Num Left:             " + str(self.num_unstarted_processes()), "blue"
            )
        )
        print(
            colored(
                "Percent Complete:     {0:.1f}%".format(self.percent_complete()), "blue"
            )
        )
        print(
            colored(
                "Time Left (D:H:M:S):  " + self.time_string(self.time_left()), "blue"
            )
        )
        print(
            colored(
                "Run Time  (D:H:M:S):  " + self.time_string(self.run_time()), "blue"
            )
        )
        for p in self.pl:
            if p.return_status == "FAIL" and p not in self.pl_errors:
                p.print_info()
                self.pl_errors.append(p)

    def start_que_runner(self):
        self.start_time = time.time()
        while True:
            # enqueu experiments
            free_gpus = self.find_free_gpu()
            for gpu in free_gpus:
                self.start_next(gpu)

            # print status
            self.update_pl_status()
            if self.print_type == "que":
                self.print_que_status()
            elif self.print_type == "process":
                if self.clear_prints:
                    os.system("clear")
                for p in self.pl:
                    p.print_info()
            else:
                raise ValueError("print type not defined: " + self.print_type)

            # check if finished
            finished = True
            failed = False
            for p in self.pl:
                if p.status != "Finished":
                    finished = False
                if p.return_status == "FAIL":
                    failed = True
            if finished:
                break
            if self.exit_on_fail and failed:
                raise RuntimeError("One or more experiements have failed")
            time.sleep(self.loop_pause)
