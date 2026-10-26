# =====================================================================================
# FILE MAP -- kvcache/radix_cache.py
# =====================================================================================
# WHAT THIS FILE IS: RADIXATTENTION'S INDEX. A tree of token sequences that remembers, for every
#                    stored beginning, WHERE (which pantry slots) its K/V notes live, so a new
#                    order that starts the same way can borrow them instead of recomputing.
#
# ELI5: Think of a phone-menu tree. Each branch is labelled with a run of words. Orders that
#       start the same way share the same branch; where they differ the tree forks.
#
#          root
#           └── [ the, cat ]  slots {10, 11}        <- shared by everyone starting "the cat"
#                 ├── [ sat ]  slots {12}           <- "the cat sat"
#                 └── [ ran ]  slots {22}           <- "the cat ran"
#
#       Each tree node holds:  key   = a run of token ids               (CPU tensor)
#                              value = the pantry slot of each of those tokens (GPU tensor)
#       So walking from the root down a path and concatenating the `value`s gives the slots of the
#       whole matched beginning (get_matched_indices).
#
# THE FIVE THINGS THE TREE DOES
#   match_prefix   walk down following the new order's tokens; report how far it got.
#                  (If the order diverges in the MIDDLE of a node, that node is SPLIT in two so the
#                  matched part becomes its own node.)
#   insert_prefix  add the rest of a finished sequence as a new leaf; report how much was already there.
#   lock/unlock    reference counts along the path: a locked path cannot be evicted.
#   evict          free the least-recently-used unlocked LEAVES (and cascade upward).
#   size_info      evictable vs protected token counts.
#
# WORKED EXAMPLE (page_size = 1)
#   1. order A "the cat sat" ids [1,2,3] finishes with slots [10,11,12]:
#         insert([1,2,3],[10,11,12]) -> root has one child: node(key [1,2,3], val [10,11,12])
#   2. order B "the cat ran" ids [1,2,4] finishes with slots [20,21,22]:
#         walk: root's child matches the first 2 tokens only -> SPLIT it at 2:
#               node(key [1,2], val [10,11]) with child node(key [3], val [12])
#         then add leaf node(key [4], val [22]) under the split node.
#         insert returns cached_len = 2: B's own slots [20,21] DUPLICATE the stored ones, so the
#         caller (CacheManager.cache_req) frees them.
#   3. a new order with ids [1,2,3,9] arrives: match_prefix([1,2,3]) -> node [3], cached_len 3,
#         slots [10,11,12]; the order only has to compute token 9.
#   4. memory runs out: evict() considers unlocked leaves [3] and [4], removes the one touched
#         LONGEST AGO, and may then remove [1,2] too if it becomes an unlocked leaf.
#
# PAGES: with page_size > 1 only whole pages are stored or matched; the first page of a key is
#       the lookup key into `children` (a tuple of page_size ids).
#
# WHO CALLS IT: scheduler/cache.py (CacheManager) via kvcache/__init__.py (name "radix").
# WHAT IT CALLS: kernel.fast_compare_key (a C++ first-difference search on CPU tensors).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

# WHAT: a function that turns the FIRST PAGE of a key into a dictionary key (see _get_key_fn).
KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


# ELI5: ONE BRANCH of the phone-menu tree.
class RadixTreeNode:
    # WHAT: class-wide counter giving each node a unique id (only used for debugging/ordering).
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        # WHAT: child branches, indexed by the first page of the child's key.
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        # ELI5: how many running orders are leaning on this branch (directly or through a
        #       descendant). 0 = free to evict (if it's a leaf).
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        # WHAT: last time this branch was touched (nanoseconds). Older = evicted first (LRU).
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        # SHAPE: _key [length] token ids (CPU); _value [length] slot numbers (GPU).
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    # WHAT: hang this branch under `parent`, registered under its first page.
    #       (If a branch with the same first page was there, it is REPLACED; split_at relies on it.)
    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    # ELI5: "How many words at the start of `input_ids` agree with this branch's words?"
    # WHAT: index of the first mismatch (or the shorter length), computed by a C++ routine.
    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    # ELI5: split a branch at position `pos`: the first `pos` words stay as a new upper branch;
    #       this node keeps the rest and hangs below it.
    #         before:  parent ── [ a b c d ]
    #         after:   parent ── [ a b ] ── [ c d ]      (split_at(2); returns the new [ a b ])
    # WHAT: the new upper node copies this node's timestamp and ref_count, because anything that
    #       leaned on the old branch also leans on its upper part.
    # NOTE: order matters: new_node.set_parent(parent) REPLACES parent's entry for this first page.
    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    # WHAT: lets nodes sit in a heap ordered by age (oldest timestamp first) for eviction.
    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


# ELI5: the TICKET for a path in the tree: how long it is, and the deepest node on it.
@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

    # WHAT: walk from the node up to the root collecting each branch's slot list, then reverse
    #       to root-first order and concatenate: the slots of the whole matched beginning.
    # SHAPE: returns [cached_len] int32 slot numbers.
    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list)


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        # WHAT: tokens stored and unlocked (can be evicted) vs stored and locked (cannot).
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

    # ELI5: pin (lock) or unpin (unlock) every branch from the ticket's node up to the root.
    # WHAT: lock: a branch going from 0 to 1 users moves from evictable to protected.
    #       unlock: a branch going from 1 to 0 users moves back. The root is never counted.
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    # ELI5: "How much of this order's beginning is already on the tree?"
    # GOTCHA: base.py says this "will not modify the cache", but _tree_walk can SPLIT a node and
    #         refresh timestamps. The stored content is unchanged; the tree shape and LRU order change.
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    # ELI5: "Write this finished beginning on the tree."
    # WHAT: 1) cut the sequence down to whole pages (insert_len = align_down(len, page_size))
    #       2) walk as far as the tree already agrees (prefix_len)
    #       3) if there's more, add ONE new leaf for the remainder, holding a CLONE of its slots
    #          (cloned so the tree owns them even if the request's page-table row changes);
    #          its tokens start out evictable (nobody has locked the new ticket yet)
    #       returns (prefix_len = how much was ALREADY there, ticket for the full inserted path).
    # WHY prefix_len matters: the caller frees its own duplicate slots for [0, prefix_len) that
    #     weren't borrowed (see scheduler/cache.py cache_req).
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    # ELI5: "Throw out the notes nobody has touched for the longest time, until `size` slots are free."
    # WHAT: only unlocked LEAVES can go (an inner branch is needed by the branches below it).
    #       A min-heap on timestamp pops the oldest leaf; its parent may become an unlocked leaf and
    #       then joins the heap. Whole nodes are removed, so more than `size` may be freed.
    #       Returns the freed slot numbers (so the clerk can reuse them).
    # RAISES: if asked for more than evictable_size, or if the heap runs dry.
    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass

    # WHAT: depth-first search for leaves nobody has locked: the eviction candidates.
    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    # ELI5: THE WALK. Start at the root and keep following the order's words down the tree.
    # WHAT: each loop turn:
    #   1) look up a child whose first page equals the order's next page. None -> stop here.
    #   2) compare the child's whole key with the order's remaining words (match_len), rounded
    #      DOWN to whole pages.
    #   3) if the child matched only partly -> SPLIT it at match_len and stop at the new upper node.
    #   4) else (fully matched) refresh its timestamp (it was just used) and go deeper.
    # RETURNS (deepest node reached, number of tokens matched).
    # GOTCHA: stamping every visited node with the SAME `tic` keeps the whole matched path equally
    #         "recent", so a path is evicted together (leaf first), never torn in the middle.
    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len


# WHAT: how to make a dictionary key out of the first page of a key.
#       page_size 1: the first token id as a plain Python int.
#       page_size N: a tuple of the first N ids (tuples are hashable).
def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
