Kubernetes manifest and the typical workflow to get everything running.

What each file does
1) infra/k8s-filter.yaml — PyTorch microservice

Defines a Deployment + Service named torch-filter.

Purpose: Hosts your PyTorch “prefilter” API (FastAPI on port 9000) that returns a probability of unsafe content for text/images.

Key bits you can tweak:

image: → container image for the microservice.

env: → where the service looks for TorchScript artifacts (IMAGE_TS, TEXT_TS), tokenizer model, and MAXLEN.

volumeMounts:/volumes: → where to mount TorchScript files (replace emptyDir with a ConfigMap, Secret, or PVC in production).

Service exposes it as an internal ClusterIP at torch-filter:9000.

2) infra/k8s-backend.yaml — FastAPI backend (OpenAI Moderation + cascade)

Creates:

ConfigMap moderation-policy (embeds policy.yaml your thresholds → actions).

Secret openai-api-key (injects OPENAI_API_KEY).

Deployment moderation-backend (your FastAPI app).

Service moderation-backend (ClusterIP on port 8000).

Purpose: Main API your frontend calls. It first tries the torch filter (early block/allow) and then calls OpenAI Moderation.

Key bits you can tweak:

image: → container image for the backend.

env: → points to TORCH_FILTER_URL (http://torch-filter:9000), policy file path, and thresholds (FILTER_*).

volumeMounts: → mounts the embedded policy from the ConfigMap at /app/policy/policy.yaml.

Secret → swap placeholder key with your real OpenAI key (or load from external secret store).

3) infra/k8s-frontend.yaml — React/Vite UI

Defines a Deployment + Service named moderation-frontend.

Purpose: Minimal UI to paste text / upload images and view moderation decisions.

Key bits you can tweak:

image: → frontend image (built from the frontend/ Dockerfile).

env: VITE_API_BASE → where the app sends API requests (defaults to http://moderation-backend:8000 inside the cluster).

Service exposes it internally as moderation-frontend (port 80 → container 5173).

Add an Ingress if you want external access via DNS/HTTP.

Typical deployment workflow
0) Prereqs

Kubernetes cluster + kubectl access.

A container registry (e.g., GHCR, ECR, GCR, Docker Hub).

1) Build & push images

From your project root:

# torch-filter
docker build -t <REG>/<NS>/torch-filter:latest torch_filter/
docker push <REG>/<NS>/torch-filter:latest

# backend
docker build -t <REG>/<NS>/moderation-backend:latest backend/
docker push <REG>/<NS>/moderation-backend:latest

# frontend (build-time arg optional to hardcode API base)
docker build -t <REG>/<NS>/moderation-frontend:latest \
  --build-arg VITE_API_BASE=http://moderation-backend:8000 \
  frontend/
docker push <REG>/<NS>/moderation-frontend:latest


Then update image: fields in the YAMLs to those pushed tags.

2) (Optional) Create a namespace
kubectl create namespace moderation
kubectl config set-context --current --namespace=moderation

3) Apply the microservice (filter) first
kubectl apply -f infra/k8s-filter.yaml
kubectl rollout status deploy/torch-filter
kubectl get svc torch-filter


If you have real TorchScript models, replace the emptyDir with a real volume or mount from a ConfigMap/Secret/PVC.

4) Apply the backend (policy + secret + API)

Edit the secret in infra/k8s-backend.yaml (or better, create it separately):

# safer: create/replace the secret without checking YAML into git
kubectl create secret generic openai-api-key \
  --from-literal=OPENAI_API_KEY="sk-..." \
  --namespace=moderation --dry-run=client -o yaml | kubectl apply -f -


Then:

kubectl apply -f infra/k8s-backend.yaml
kubectl rollout status deploy/moderation-backend
kubectl get svc moderation-backend

5) Apply the frontend
kubectl apply -f infra/k8s-frontend.yaml
kubectl rollout status deploy/moderation-frontend
kubectl get svc moderation-frontend

6) Access the app

Without Ingress: port-forward services for quick testing:

kubectl port-forward svc/moderation-frontend 8080:80
# open http://localhost:8080


The frontend talks to the backend via the internal service name http://moderation-backend:8000.

With Ingress: create an Ingress resource (example was provided earlier), then map DNS to your ingress controller LB IP.

Verifying each component
Torch-filter health
kubectl port-forward svc/torch-filter 9000:9000
curl http://localhost:9000/health
# => {"ok":true,"image_model":<bool>,"text_model":<bool>}

Backend health
kubectl port-forward svc/moderation-backend 8000:8000
curl http://localhost:8000/api/health
# => {"ok": true}

End-to-end call (text)
curl -s -X POST http://localhost:8000/api/moderate/text \
  -H "Content-Type: application/json" \
  -d '{"text":"Have a great day!"}' | jq

End-to-end call (image)
curl -s -X POST http://localhost:8000/api/moderate/image \
  -F "image_url=https://picsum.photos/200" | jq

Common tweaks

Mounting real models: Replace in k8s-filter.yaml:

volumes:
  - name: filters
    emptyDir: {}


with a PVC or ConfigMap/Secret containing your model.ts and text_model.ts. Keep mountPath: /app/filters.

Changing policy at runtime: Update the ConfigMap then roll the backend:

kubectl apply -f infra/k8s-backend.yaml
kubectl rollout restart deploy/moderation-backend


Updating images (new code):

kubectl set image deploy/torch-filter torch-filter=<REG>/<NS>/torch-filter:<tag>
kubectl set image deploy/moderation-backend backend=<REG>/<NS>/moderation-backend:<tag>
kubectl set image deploy/moderation-frontend frontend=<REG>/<NS>/moderation-frontend:<tag>


External access: Add an Ingress (or use a LoadBalancer Service) for the frontend. Keep the backend ClusterIP and call it from the frontend via http://moderation-backend:8000.