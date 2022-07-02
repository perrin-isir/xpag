# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import os
import logging
from typing import Union, Any
from xpag.agents.agent import Agent

global_first_eval_log_done = None
global_eval_logger = None


class LevelFilter(logging.Filter):
    def __init__(self, level):
        super().__init__()
        self.__level = level

    def filter(self, logrecord):
        return logrecord.levelno == self.__level


def eval_log_reset():
    global global_first_eval_log_done, global_eval_logger
    if global_eval_logger is not None:
        for handler in global_eval_logger.handlers[:]:
            global_eval_logger.removeHandler(handler)
            handler.close()
    global_first_eval_log_done = None
    global_eval_logger = None


def eval_log(
    steps: int,
    elapsed_time: float,
    reward: float,
    is_success: Union[float, None],
    env_info: Any,
    agent: Agent,
    save_dir: Union[str, None] = None,
):
    global global_first_eval_log_done, global_eval_logger
    if global_first_eval_log_done is None:
        global_first_eval_log_done = True
        if save_dir:
            s_dir = os.path.expanduser(save_dir)
            os.makedirs(s_dir, exist_ok=True)
            print("Logging in " + s_dir)
            open(os.path.join(s_dir, "log.txt"), "w").close()
            with open(os.path.join(s_dir, "config.txt"), "w") as f:
                print("env_info:", file=f)
                for key, elt in env_info.items():
                    print(f"{key}: {elt}", file=f)
                print("\nagent config:", file=f)
                agent.write_config(f)
                f.close()
        global_eval_logger = logging.getLogger("eval_log")
        global_eval_logger.setLevel(logging.DEBUG)
        global_eval_logger.propagate = False
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        chformatter = logging.Formatter("%(message)s")
        ch.setFormatter(chformatter)
        global_eval_logger.addHandler(ch)
        if save_dir:
            fh = logging.FileHandler(
                os.path.join(os.path.expanduser(save_dir), "log.txt")
            )
            fh.setLevel(logging.INFO)
            fhfilter = LevelFilter(logging.INFO)
            fh.addFilter(fhfilter)
            fhformatter = logging.Formatter("%(message)s")
            fh.setFormatter(fhformatter)
            global_eval_logger.addHandler(fh)
        if is_success is not None:
            init_list = [
                "steps",
                "delta_training_time_ms",
                "episode_reward",
                "success_at_the_end",
            ]
        else:
            init_list = ["steps", "delta_training_time_ms", "episode_reward"]
        global_eval_logger.info(",".join(map(str, init_list)))
    message_string = (
        f"[{steps:12} steps] [training time (ms) +="
        f" {elapsed_time:<10.0f}] [ep reward: {reward:<15.3f}] "
    )
    log_info = [steps, elapsed_time, reward]
    if is_success is not None:
        message_string += f"[success: {is_success:<3.2f}]"
        log_info.append(is_success)
    global_eval_logger.warning(message_string)
    if save_dir:
        global_eval_logger.info(",".join(map(str, log_info)))
