import os
import numpy as np
import time
import gym
from IPython import embed

if True:
    """ 
    avoid mujoco rendering bug
    """
    import sys
    import subprocess

    preload = os.environ.get("LD_PRELOAD", "")
    if os.environ["HOME"] == os.path.expanduser("~"):
        if not preload:
            to_preload = "/usr/lib/x86_64-linux-gnu/libGLEW.so"
            os.environ["LD_PRELOAD"] = to_preload
            print("Restarting with LD_PRELOAD={0}".format(to_preload))
            os.execv(sys.executable, [sys.executable] + sys.argv)
    glxinfo = subprocess.Popen("glxinfo", stdout=subprocess.PIPE)
    output_glxinfo = glxinfo.communicate()[0]
    for line in output_glxinfo.decode("utf-8").split("\n"):
        if (
            "GLX version" in line
            or "OpenGL vendor string" in line
            or "OpenGL renderer string" in line
            or "OpenGL core profile version" in line
        ):
            print(line)

prefix = os.path.join(os.path.expanduser("~"), "results", "xpag")
dirfiles = os.listdir(prefix)
dirfiles.sort()
load_dir = os.path.join(prefix, dirfiles[-1])

env_name = str(np.loadtxt(os.path.join(load_dir, 'episode', 'env_name.txt'),
                          dtype='str'))
env = gym.make(env_name)
qpos = np.load(os.path.join(load_dir, 'episode', 'qpos.npy'))
qvel = np.load(os.path.join(load_dir, 'episode', 'qvel.npy'))

i = 0
while True:
    tic = time.time()
    env.set_state(qpos[i], qvel[i])
    env.render()
    i = (i+1) % len(qpos)
    toc = time.time()
    elapsed = toc - tic
    # 60 Hz = normal speed
    dt_sleep = max((0, 1.0 / 60.0 - elapsed))
    time.sleep(dt_sleep)
