# Tennis AI Coach

Tennis swing analysis app with:
- `frontend`: React + Vite UI (record from webcam or upload video)
- `backend`: FastAPI + MediaPipe/OpenCV analysis + Amazon Nova coaching feedback

## Features
- Record swing video directly in browser
- Upload local video files (`mp4`, `mov`, `webm`, etc.)
- Analyze `forehand`, `backhand`, and `serve`
- Show metrics, flags, camera-angle estimate, and coaching guidance
- Human-readable feedback cards in frontend (not raw JSON)

## Project Structure
- `frontend/` React app
- `backend/` FastAPI app and analysis logic
- `test_videos/` local sample videos (do not commit large files)

## Requirements
- Node.js 18+
- Python 3.10+
- macOS/Linux recommended for local CV dependencies

## Run Locally

### 1) Start Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend endpoints:
- `GET /health`
- `POST /analyze`

### 2) Start Frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

Vite proxies `/api/*` to backend (`http://127.0.0.1:8000`).

## Environment Variables (Optional)
Create `backend/.env` if using Nova feedback:

```env
AWS_REGION=us-east-1
NOVA_MODEL_ID=amazon.nova-lite-v1:0
```

You also need valid AWS credentials for Bedrock in your local environment.

## Notes on Video Quality
For better pose detection:
- Keep full body in frame
- Use good lighting
- Prefer side-view camera angle

## GitHub Large File Limit
GitHub rejects files larger than 100 MB. Do not commit raw videos.

Recommended `.gitignore` entries:
```gitignore
test_videos/
*.mp4
*.mov
*.webm
```

If you accidentally committed a large video, remove it from history before pushing.

## Build Checks
Frontend:
```bash
cd frontend
npm run build
```

Backend syntax check:
```bash
python3 -m py_compile backend/main.py
```
