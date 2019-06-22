import furry.utils
import furry.init
import furry.data
from furry.data import float32, float64, float16, uint8, int8, int16, int32, int64, tensor, Tensor
import furry.dev
from furry.dev import device, cpu, gpu
import furry.activation
import furry.loss
import furry.optimizer
import furry.logger
from furry.session import session
import furry.module
from furry.module import Module
import furry.nn
from furry.__modularized import Flatten

VERSION = "1.1.0"

def print_stat():
    import platform
    print("FURRY ENVIRONMENT STAT")
    print("  FURRY")
    print("    VERSION: %s" % (VERSION,))
    print("    BRANCH: dev")
    print("  CPU: %s" % (platform.processor(),))
    print("  GPU(CUDA)")
    print("    AVAILABLE: %s" % (str(furry.dev.CUDA_AVAILABLE).lower(),))
    if furry.dev.CUDA_AVAILABLE:
        print("    %i DEVICES" % (furry.dev.CUDA_DEVICE_COUNT,))
        for cuda_device_id in range(furry.dev.CUDA_DEVICE_COUNT):
            print("    GPU %i, %s" % (cuda_device_id, furry.dev.gpu_device_name(cuda_device_id),))
