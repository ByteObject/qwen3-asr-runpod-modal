import runpod

from core import handler

runpod.serverless.start({"handler": handler})
