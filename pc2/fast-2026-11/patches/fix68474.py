#!/usr/bin/env python3
"""fix68474.py ROOT : hand-apply the 4 hunks of vllm#68474 (stack) that do not apply on 8586ccb."""
import sys, os

root = sys.argv[1]

p = os.path.join(root, "vllm/model_executor/kernels/linear/__init__.py")
s = open(p).read()
a = "import vllm.envs as envs\nfrom vllm.logger import init_logger\n"
assert s.count(a) == 1, "import anchor"
s = s.replace(a, "import vllm.envs as envs\nfrom vllm.config.quantization import PhaseSpec, QuantSpec\nfrom vllm.logger import init_logger\n", 1)
a = '        logger.debug_once("Using %s for MXFP8 GEMM", kernel_cls.__name__)\n        return kernel_cls(config)\n'
assert s.count(a) == 1, "mxfp8 return anchor"
s = s.replace(a, '''        logger.debug_once("Using %s for MXFP8 GEMM", kernel_cls.__name__)
        return maybe_phase_split(
            kernel_cls(config),
            # Batched (BMM) layers keep their own activations.
            None if bmm_batch_size is not None else _user_linear_phase("mxfp8"),
            weight="mxfp8",
            candidates=_POSSIBLE_MXFP8_KERNELS.get(platform, ()),
        )
''', 1)
open(p, "w").write(s)

p = os.path.join(root, "vllm/model_executor/model_loader/utils.py")
s = open(p).read()
a = "    finalize_model_weights(model, target_device)\n"
assert s.count(a) == 1, "finalize anchor"
s = s.replace(a, a + "    check_linear_phase_served(model)\n", 1)
open(p, "w").write(s)
print("ok")
