#!/bin/bash
echo "==> Stopping all containers..."
podman pod stop rhel-ai-pod
podman pod rm rhel-ai-pod
echo "==> Done. vectorstore data preserved in ~/vectorstore"
