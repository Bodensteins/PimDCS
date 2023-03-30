import numpy as np
from enum import IntEnum


class Archid:
    def __init__(self, *args):
        if len(args) < 5:
            args = list(args).extend([0] * (5 - len(args)))
        self.crx_id, self.pe_id, self.tile_id, self.bank_id, self.chip_id = args

    def get_arch_id_list(self):
        return [self.crx_id, self.pe_id, self.tile_id, self.bank_id, self.chip_id]

    def __str__(self):
        return "[crx_id, pe_id, tile_id, bank_id, chip_id] = [{}, {}, {}, {}, {}]".format(*self.get_arch_id_list())


class ArchLevel(IntEnum):
    crx_level = 1
    pe_level = 2
    tile_level = 3
    bank_level = 4
    chip_level = 5


class usageStatus(IntEnum):
    free = 0  # all free
    full = 1  # all used
    used = 2  # partial used


class segmentTree:
    def __init__(self, size):
        self.size = size
        self.node_value = np.zeros(size * 4 + 5, dtype=np.int32)
        self.node_lazy = np.zeros(size * 4 + 5, dtype=np.int8)

    # value only support 0 and 1
    def set(self, setL, setR, value):
        assert (value == 0 or value == 1)
        assert (0 <= setL <= setR < self.size)
        self._set(1, 0, self.size - 1, setL, setR, value)

    def _down_lazy_node(self, x, l, r):
        if self.node_lazy[x] == 0:
            return
        elif self.node_lazy[x] == -1:
            self.node_value[x] = 0
        else:
            self.node_value[x] = (r - l + 1)

        if l != r:
            self.node_lazy[x * 2] = self.node_lazy[x]
            self.node_lazy[x * 2 + 1] = self.node_lazy[x]
        self.node_lazy[x] = 0

    # value should not be 0 or 1
    def _set(self, x, l, r, setL, setR, value):
        if setL <= l and setR >= r:
            if value == 0:
                self.node_lazy[x] = -1
            else:
                self.node_lazy[x] = 1
            self._down_lazy_node(x, l, r)
            return

        self._down_lazy_node(x, l, r)
        mid = (l + r) // 2
        # (l...mid)  (mid+1...r)
        if setL <= mid:
            self._set(x * 2, l, mid, setL, setR, value)
        if setR > mid:
            self._set(x * 2 + 1, mid + 1, r, setL, setR, value)
        self._down_lazy_node(x * 2, l, mid)
        self._down_lazy_node(x * 2 + 1, mid + 1, r)
        self.node_value[x] = self.node_value[x * 2] + self.node_value[x * 2 + 1]

    def query(self, qL, qR):
        assert (0 <= qL <= qR < self.size)
        return self._query(1, 0, self.size - 1, qL, qR)

    def _query(self, x, l, r, qL, qR):
        self._down_lazy_node(x, l, r)
        if qL <= l and qR >= r:
            return self.node_value[x]
        mid = (l + r) // 2
        ansl, ansr = 0, 0
        if qL <= mid:
            ansl = self._query(x * 2, l, mid, qL, qR)
        if qR > mid:
            ansr = self._query(x * 2 + 1, mid + 1, r, qL, qR)
        return ansl + ansr
