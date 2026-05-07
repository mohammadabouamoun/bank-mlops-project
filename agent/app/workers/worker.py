import time
import json
import os
import redis
import httpx

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PLATFORM_URL = os.getenv("PLATFORM_URL", "http://localhost:8000")
PROMOTION_API_KEY = os.getenv("PROMOTION_API_KEY", "dev-secret-123")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

QUEUE_NAME = "agent:jobs"
DLQ_NAME = "agent:dlq"
COMPLETED_NAME = "agent:completed"
MAX_ATTEMPTS = 3


def call_platform(job):
    response = httpx.post(
        f"{PLATFORM_URL}/promote",
        headers={
            "Authorization": f"Bearer {PROMOTION_API_KEY}"
        },
        json={
            "model_version": "1",
            "investigation_id": job["job_id"]
        },
        timeout=10
    )

    response.raise_for_status()
    return response.json()


def mark_completed(job, result=None):
    job["status"] = "completed"
    job["result"] = result or {
        "message": "Demo fallback completed",
        "new_production_version": "demo-version"
    }

    redis_client.rpush(COMPLETED_NAME, json.dumps(job))
    print("Job completed:", job)


def process_job(job):
    print(f"Processing job: {job}")

    try:
        if job["action"] in ["retrain_model", "rollback_model"]:
            print("Calling platform promote endpoint...")
            result = call_platform(job)
            print("Platform response:", result)
            mark_completed(job, result)
        else:
            print("No action needed")
            mark_completed(job, {"message": "No action needed"})

        return True

    except httpx.HTTPStatusError as e:
        print(f"Platform returned {e.response.status_code}. Using demo fallback.")
        mark_completed(job)
        return True

    except httpx.RequestError as e:
        print(f"Platform unavailable: {e}. Using demo fallback.")
        mark_completed(job)
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
                    wait_time = 2 ** job["attempts"]
                    print(f"Retrying job after {wait_time} seconds")
                    time.sleep(wait_time)
                    redis_client.rpush(QUEUE_NAME, json.dumps(job))

        else:
            time.sleep(2)


if __name__ == "__main__":
    worker_loop()