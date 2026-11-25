# Alchemy state

This branch holds the Alchemy state of this repository. Alchemy deploys write it; do not edit it
by hand. Each resource is one file, `{stack}/{stage}/{resource}.json`.

Each secret value is `{"__redacted__": {"__sealed__": "v1.<iv>.<ciphertext>"}}`, sealed with
AES-256-GCM under the key in ALCHEMY_GITHUB_STATE_KEY. Every other value is plain JSON.

The deploy lock of each stage is on the branch of the same name with `-lock` at the end.
