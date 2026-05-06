import json
import os
import redis
from uuid import uuid4

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

QUEUE_NAME = "agent:jobs"
DLQ_NAME = "agent:dlq"


def add_to_queue(action: str, investigation_id: str):
    job = {
        "job_id": str(uuid4()),
        "action": action,
        "idempotency_key": investigation_id,
        "status": "queued",
        "attempts": 0
    }

    redis_client.rpush(QUEUE_NAME, json.dumps(job))
    return job


def get_queue():
    jobs = redis_client.lrange(QUEUE_NAME, 0, -1)
    return [json.loads(job) for job in jobs]


def get_dlq():
    jobs = redis_client.lrange(DLQ_NAME, 0, -1)
    return [json.loads(job) for job in jobs]