import jax
import jax.numpy as jnp


@jax.jit
def my_jitted_function(x, y):
    return jnp.dot(x, y)


def parent_function(x, y):
    my_jitted_function(x, y)


def launch_test(backend):
    a = jnp.ones(2)
    jnp.dot(a, a)
    print(jax.devices(backend))
    a = jax.device_put(jnp.ones(200), device=jax.devices(backend)[0])
    parent_function(a, a)
    with jax.profiler.trace(f"output/debug_{backend}"):
        for _ in range(1_000):
            parent_function(a, a)


launch_test("cpu")
launch_test("gpu")
