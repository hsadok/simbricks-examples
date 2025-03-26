# This program reads a csv with memory accesses and groups accesses into
# different memory regions

import csv
import sys

from collections import defaultdict

from sortedcontainers import SortedList


def overlap(region1, region2):
    return region1[0] <= region2[1] and region1[1] >= region2[0]


class RangeTree(SortedList):
    def __init__(self, iterable=None, key=None):
        super().__init__(iterable=iterable, key=key)

    def add_range(self, start, end):
        new_range = (start, end)

        start = self.bisect_right(new_range)
        start = max(start - 1, 0)

        removals = []
        for r in self.islice(start):
            if new_range[1] < r[0]:
                break
            if overlap(r, new_range):
                removals.append(r)
                new_range = (min(r[0], new_range[0]), max(r[1], new_range[1]))

        for rem in removals:
            self.remove(rem)

        self.add(new_range)

    def __add__(self, other):
        copy = self.copy()
        for range in other:
            copy.add_range(range[0], range[1])
        return copy

    def __contains__(self, value):
        if len(self) == 0:
            return False

        left = self.bisect_right((value, value))
        left = max(left - 1, 0)
        right = min(left + 1, len(self) - 1)
        return overlap(self[left], (value, value)) or overlap(
            self[right], (value, value)
        )


def group_memory_accesses(memory_accesses):
    regions = {
        "read": RangeTree(),
        "write": RangeTree(),
    }

    for access in memory_accesses:
        access_type = access["access_type"]
        address = access["address"]
        size = access["size"]

        region = regions[access_type]
        region.add_range(address, address + size)

    return regions


def save_accesses_by_address(memory_accesses, filename, access_type=None):
    addresses = defaultdict(list)
    for count, access in enumerate(memory_accesses):
        if access_type is not None and access["access_type"] != access_type:
            continue
        for addr in range(
            access["address"], access["address"] + access["size"], 4
        ):
            addresses[addr].append(count)

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        for addr, access_indices in sorted(addresses.items()):
            writer.writerow([addr] + access_indices)


def save_access_count_by_address(memory_accesses, filename, access_type=None):
    addresses = defaultdict(int)
    for count, access in enumerate(memory_accesses):
        if access_type is not None and access["access_type"] != access_type:
            continue
        for addr in range(
            access["address"], access["address"] + access["size"], 4
        ):
            addresses[addr] += 1

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        for addr, access_count in sorted(addresses.items()):
            writer.writerow([addr, access_count])


def read_memory_accesses(filename):
    memory_accesses = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            access = {
                "access_type": row[0],
                "address": int(row[1]),
                "size": int(row[2]),
            }
            memory_accesses.append(access)
    return memory_accesses


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <input_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]

    memory_accesses = read_memory_accesses(input_csv)
    save_accesses_by_address(
        memory_accesses,
        "memory_accesses_by_address_read.csv",
        access_type="read",
    )
    save_accesses_by_address(
        memory_accesses,
        "memory_accesses_by_address_write.csv",
        access_type="write",
    )
    save_access_count_by_address(
        memory_accesses,
        "memory_access_count_by_address_read.csv",
        access_type="read",
    )
    save_access_count_by_address(
        memory_accesses,
        "memory_access_count_by_address_write.csv",
        access_type="write",
    )
    regions = group_memory_accesses(memory_accesses)

    regions["both"] = regions["read"] + regions["write"]

    for access_type, regions in regions.items():
        total_bytes = sum([r[1] - r[0] for r in regions])
        print(f"{access_type.capitalize()} Regions ({total_bytes} bytes):")
        for region in regions:
            start, end = region
            print(f"  {start:#018x} - {end:#018x} ({end - start} bytes)")
        print()


if __name__ == "__main__":
    main()
