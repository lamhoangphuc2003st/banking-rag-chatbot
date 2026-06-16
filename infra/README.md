# Infrastructure

This folder contains deployment assets for the production rebuild:

- `k8s/`: starter Kubernetes manifests for API and web.
- `prometheus/`: local Prometheus scrape config.

Secrets such as API keys, database URLs, Redis URLs, and Qdrant keys must be created as Kubernetes Secrets or managed by a cloud secret manager. Do not commit production secrets.

The API deployment expects a read-only PVC named `bankbot-rag-data` mounted at `/app/data`.
Populate it with generated `data/normalized` and `data/chunks` artifacts before rolling out the API, or configure `RAG_DATA_ROOT` to another mounted artifact location.

## Managed Redis

Use one managed Redis instance for both local development and production instead
of running Redis inside Docker Compose or inside the API pod. The API reads the
managed Redis endpoint from `REDIS_URL`.

For local development, put the provider connection string in `.env`. For
production, put the same connection string, or another database on the same
managed Redis service, in the `bankbot-api-secrets` Kubernetes Secret.

Use `rediss://` if the provider requires TLS, for example:

```text
rediss://default:<password>@<managed-redis-host>:6380/0
```

The same Redis connection is used for:

- API rate limiting, enabled with `API_RATE_LIMIT_BACKEND=redis`.
- RAG cache, enabled with `RAG_CACHE_BACKEND=redis`.

Create the real secret from your provider credentials, or copy
`k8s/secrets.example.yaml` and replace every `CHANGE_ME` value before applying it.
