# 🎾 Tennis AI Coach (Demo)

A hackathon demo that records a tennis swing in the browser, uploads the video to a FastAPI backend, and (next) runs pose analysis + Nova coaching feedback.

## Tech Stack
- Frontend: React + Vite (TSX)
- Backend: FastAPI
- CV: MediaPipe Pose + OpenCV (planned)
- LLM: Amazon Nova via Bedrock (planned)

## Run Locally

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn python-multipart
uvicorn main:app --reload