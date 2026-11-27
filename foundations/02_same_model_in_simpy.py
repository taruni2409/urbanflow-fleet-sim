"""Day 1b - The exact same model in SimPy.

Compare with a_scheduler_by_hand.py:
  heap of events + clock      -> simpy.Environment
  free_vehicles + waiting     -> simpy.Resource(capacity=N_VEHICLES)
  schedule(now + d, "finish") -> yield env.timeout(d)
The results must be identical. If they are not, one of the two is wrong.
"""
import simpy

ARRIVALS = [0, 2, 3, 7, 8]
DURATIONS = [6, 9, 4, 5, 3]
N_VEHICLES = 2

start = {}


def passenger(env, p, fleet):
    yield env.timeout(ARRIVALS[p])            # wait until my request time
    print(f"t={env.now:>2}  P{p} requests a ride   (queue={len(fleet.queue)})")
    with fleet.request() as car:               # join the FIFO queue for a car
        yield car                              # paused here until a car is free
        start[p] = env.now
        print(f"t={env.now:>2}  P{p} picked up         (ends at t={env.now + DURATIONS[p]})")
        yield env.timeout(DURATIONS[p])        # the trip itself
    print(f"t={env.now:>2}  P{p} trip finished")  # leaving 'with' releases the car


env = simpy.Environment()
fleet = simpy.Resource(env, capacity=N_VEHICLES)
for p in range(len(ARRIVALS)):
    env.process(passenger(env, p, fleet))
env.run()

waits = [start[p] - ARRIVALS[p] for p in range(len(ARRIVALS))]
print("\nwaits:", waits, " average:", sum(waits) / len(waits))
