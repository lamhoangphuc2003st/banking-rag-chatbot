# Infrastructure

This folder contains deployment assets for the production rebuild:

- `k8s/`: starter Kubernetes manifests for API and web.
- `prometheus/`: local Prometheus scrape config.

Secrets such as API keys, database URLs, Redis URLs, and Qdrant keys must be created as Kubernetes Secrets or managed by a cloud secret manager. Do not commit production secrets.
