"""RunPod Serverless Handler for Qwen3-ASR"""

import runpod


def handler(event):
    """RunPod serverless handler entry point."""
    # Lazy import to speed up cold start detection
    from core import process_request, sanitize_error

    try:
        job_input = event["input"]
        if not job_input:
            return {"error": "No input provided"}
        return process_request(job_input)
    except Exception as e:
        return {"error": sanitize_error(e)}


runpod.serverless.start({"handler": handler})
