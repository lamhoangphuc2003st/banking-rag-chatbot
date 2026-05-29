# Data Governance

## Source Policy

The project only processes public pages from Vietcombank. The crawler must respect `robots.txt`, rate limits, and source attribution. Private APIs, authenticated pages, personal data, cookies, forms, and user account information are out of scope.

## Required Metadata

Each normalized document must include:

- `source_url`
- `title`
- `content_hash`
- `crawl_time`
- `language`
- `product_type`
- `section`
- `raw_artifact_path`

## Quality Checks

- Reject documents with empty content, missing URL, missing title, or unsupported language.
- Deduplicate by normalized content hash.
- Record content diffs between crawl versions.
- Keep raw artifacts for auditability.

## Privacy

- Do not ask users for passwords, OTPs, card numbers, full account numbers, or identity documents.
- Redact accidental PII from logs.
- Store conversation data only for observability and evaluation with clear retention settings.
