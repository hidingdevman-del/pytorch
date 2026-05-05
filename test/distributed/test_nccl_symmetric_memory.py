# Owner(s): ["module: c10d"]

# Tests the NCCL symm_mem backend — `ncclCommWindowRegister` inside
# NCCLPeerAllocInfo. Reproduces a regression where the call returns NCCL
# `invalid argument` (NCCL 2.29.7) on a sub-group whose size equals WORLD.
#
# Lives in its own file because importing test_symmetric_memory.py at module
# load locks the symm_mem allocator's `in_use_` flag, which makes
# `set_backend("NCCL")` fail. Keeping imports here minimal avoids that.
from unittest import skipIf

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    PLATFORM_SUPPORTS_SYMM_MEM,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    requires_cuda_p2p_access,
    run_tests,
)


device_type = "cuda"
device_module = torch.get_device_module(device_type)


@requires_cuda_p2p_access()
class NCCLSymmetricMemoryTest(MultiProcessTestCase):
    """`set_backend` is one-shot per process, so we use MultiProcessTestCase
    to spawn a fresh process for the test."""

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return device_module.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _init_process(self) -> None:
        # set_backend FIRST — once a symm_mem allocator is touched (e.g.
        # via symm_mem.empty) the choice is locked.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        # device_id ensures the default PG is constructed with
        # bound_device_id, which (1) eagerly connects WORLD's NCCL comm and
        # (2) makes new_group take the split-from path so subgroups also
        # eagerly init. Without this, getCommPtr() returns NULL and
        # ncclCommWindowRegister fails with `comm argument is NULL`.
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=self.device,
        )

    @skipIf(
        not PLATFORM_SUPPORTS_SYMM_MEM, "SymmMem is not supported on this ROCm arch"
    )
    @skip_if_lt_x_gpu(2)
    def test_world(self) -> None:
        """Rendezvous on the default WORLD process group on the NCCL backend.
        Baseline that doesn't go through new_group / split-comm — useful to
        tell whether failures are sub-group-specific or hit the WORLD path
        too."""
        self._init_process()

        t = symm_mem.empty(64, device=self.device)
        symm_mem_world = symm_mem.rendezvous(t, group=dist.group.WORLD)

        self.assertEqual(symm_mem_world.world_size, self.world_size)
        self.assertEqual(symm_mem_world.rank, self.rank)

        t.fill_(self.rank)
        # NCCLSymmetricMemory::barrier is NYI on this backend; use the
        # regular collective barrier to synchronize after fill.
        dist.barrier()

        peer_rank = (self.rank + 1) % self.world_size
        buf_world = symm_mem_world.get_buffer(peer_rank, (64,), torch.float32)
        self.assertTrue(buf_world.eq(peer_rank).all())

    @skipIf(
        not PLATFORM_SUPPORTS_SYMM_MEM, "SymmMem is not supported on this ROCm arch"
    )
    @skip_if_lt_x_gpu(2)
    def test_subgroup(self) -> None:
        """Rendezvous on a sub-group whose ranks equal WORLD's, on the NCCL
        backend. Hits ncclCommWindowRegister inside NCCLPeerAllocInfo."""
        self._init_process()

        ranks = list(range(self.world_size))
        subgroup = dist.new_group(ranks)

        t = symm_mem.empty(64, device=self.device)
        symm_mem_subgroup = symm_mem.rendezvous(t, group=subgroup)

        self.assertEqual(symm_mem_subgroup.world_size, self.world_size)
        self.assertEqual(symm_mem_subgroup.rank, self.rank)

        t.fill_(self.rank)
        # NCCLSymmetricMemory::barrier is NYI on this backend; use the
        # regular collective barrier on the subgroup.
        dist.barrier(group=subgroup)

        peer_rank = (self.rank + 1) % self.world_size
        buf_sub = symm_mem_subgroup.get_buffer(peer_rank, (64,), torch.float32)
        self.assertTrue(buf_sub.eq(peer_rank).all())


if __name__ == "__main__":
    run_tests()
