import sys
import os
import time
import glob
import subprocess
from termcolor import colored


class Process:
    def __init__(self, name, cmd, cwd="./"):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.status = "Not Started"
        self.gpu = -1
        self.process = None
        self.return_status = "NONE"
        self.run_time = 0

    def start(self, gpu=0):
        with open(os.devnull, "w") as devnull:
            self.process = subprocess.Popen(
                self.cmd.split(" "),
                cwd=self.cwd,
                stdout=devnull,
                stderr=subprocess.PIPE,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
            )

        self.pid = self.process.pid

        self.status = "Running"
        self.start_time = time.time()
        self.gpu = gpu

    def update_status(self):
        if self.status == "Running":
            self.run_time = time.time() - self.start_time
            if self.process.poll() is not None:
                self.status = "Finished"
                if self.process.poll() == 0:
                    self.return_status = "SUCCESS"
                else:
                    self.return_status = "FAIL"
                    self.return_code = self.process.returncode
                    output, error = self.process.communicate()
                    if not error is None:
                        self.error = error.decode("utf-8")
                    else:
                        self.error = "No error message. :("

    def get_pid(self):
        return self.pid

    def get_status(self):
        return self.status

    def get_gpu(self):
        return self.gpu

    def get_return_status(self):
        return self.return_status

    def print_info(self):
        print_string = "\n" + colored(f"Process info for: {self.name}", "blue") + "\n"
        print_string = print_string + colored("cmd is ", "blue") + self.cmd + "\n"
        print_string = print_string + (colored("cwd is ", "blue") + self.cwd + "\n")
        print_string = print_string + (colored("status ", "blue") + self.status + "\n")
        if self.return_status == "SUCCESS":
            print_string = print_string + (
                colored("return status ", "blue")
                + colored(self.return_status, "green")
                + "\n"
            )
        elif self.return_status == "FAIL":
            print_string = print_string + (
                colored("return status ", "blue")
                + colored(self.return_status, "red")
                + "\n"
                + colored("START OF ERROR MESSAGE", "red")
                + "\n"
                + self.error
                + colored("END OF ERROR MESSAGE", "red")
                + "\n"
            )
        else:
            print_string = print_string + (
                colored("return status ", "blue")
                + colored(self.return_status, "yellow")
                + "\n"
            )

        print_string = print_string + (
            colored("run time ", "blue") + str(self.run_time)
        )
        print(print_string)
