"""Day 1a - A discrete-event simulator built by hand.

This is what SimPy does internally: keep a priority queue of future
events ordered by time, pop the earliest, jump the clock to it, apply it,
and schedule any new events it causes.

Model: 2 vehicles, 5 passengers, first-come-first-served.
"""
import heapq

ARRIVALS = [0, 2, 3, 7, 8]      # minute each passenger requests a ride
DURATIONS = [6, 9, 4, 5, 3]     # trip length in minutes
N_VEHICLES = 2

events = []                      # heap of (time, seq, kind, passenger)
seq = 0                          # tie-breaker so equal times keep insert order


def schedule(time, kind, passenger):
    global seq
    heapq.heappush(events, (time, seq, kind, passenger))
    seq += 1


for p, t in enumerate(ARRIVALS):
    schedule(t, "arrive", p)

free_vehicles = N_VEHICLES       # STATE: how many cars are idle
waiting = []                     # STATE: FIFO queue of passengers
start = {}                       # OUTPUT: when each trip started

while events:
    now, _, kind, p = heapq.heappop(events)   # clock JUMPS to next event
    if kind == "arrive":
        waiting.append(p)
        print(f"t={now:>2}  P{p} requests a ride   (queue={waiting})")
    else:
        free_vehicles += 1
        print(f"t={now:>2}  P{p} trip finished     (free cars={free_vehicles})")

    # Dispatch rule: while a car is free and someone waits, start a trip.
    while free_vehicles and waiting:
        q = waiting.pop(0)
        free_vehicles -= 1
        start[q] = now
        schedule(now + DURATIONS[q], "finish", q)
        print(f"t={now:>2}  P{q} picked up         (ends at t={now + DURATIONS[q]})")

waits = [start[p] - ARRIVALS[p] for p in range(len(ARRIVALS))]
print("\nwaits:", waits, " average:", sum(waits) / len(waits))
