import time
import json
import os
import redis
import httpx

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PLATFORM_URL = os.getenv("PLATFORM_URL", "http://localhost:8000")
PROMOTION_API_KEY = os.getenv("PROMOTION_API_KEY", "test-token")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

QUEUE_NAME = "agent:jobs"
DLQ_NAME = "agent:dlq"
MAX_ATTEMPTS = 3


def call_platform(job):
    response = httpx.post(
        f"{PLATFORM_URL}/promote",
        headers={
            "Authorization": f"Bearer {PROMOTION_API_KEY}"
        },
        json={
            "model_version": "v2",
            "investigation_id": job["job_id"]
        },
        timeout=10
    )

    response.raise_for_status()
    return response.json()


def process_job(job):
    print(f"Processing job: {job}")

    try:
        if job["action"] in ["retrain_model", "rollback_model"]:
            print("Calling platform promote endpoint...")
            result = call_platform(job)
            print("Platform response:", result)
        else:
            print("No action needed")

        return True

    except Exception as e:
        print(f"Job failed: {e}")
        return False


def worker_loop():
    print("Worker started...")

    while True:
        job_data = redis_client.lpop(QUEUE_NAME)

        if job_data:
            job = json.loads(job_data)

            success = process_job(job)

            if not success:
                job["attempts"] += 1

                if job["attempts"] >= MAX_ATTEMPTS:
                    job["status"] = "failed"
                    redis_client.rpush(DLQ_NAME, json.dumps(job))
                    print("Moved to DLQ")
                else:
                    job["status"] = "retrying"
                    redis_client.rpush(QUEUE_NAME, json.dumps(job))
                    print("Retrying job")

        else:
            time.sleep(2)


if __name__ == "__main__":
    worker_loop()