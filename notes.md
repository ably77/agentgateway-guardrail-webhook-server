docker buildx rm ly-builder
docker buildx create --use --name ly-builder
docker buildx inspect ly-builder --bootstrap

docker buildx build --platform linux/amd64,linux/arm64 -t ably7/ai-guardrail-webhook-server:0.1.1 --push .

docker push ably7/ai-guardrail-webhook-server:0.1.1