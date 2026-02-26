import { useEffect, useRef, useState } from "react";

type UploadResponse = {
  saved: string;
};

export default function App() {
  const liveVideoRef = useRef<HTMLVideoElement | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [recording, setRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [uploading, setUploading] = useState(false);
  const [savedName, setSavedName] = useState<string>("");

  // Automatically start camera on load (optional)
  useEffect(() => {
    void startCamera();
    return () => stopCamera();
  }, []);

  async function startCamera() {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, frameRate: 30 },
        audio: false,
      });

      setStream(mediaStream);

      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = mediaStream;
        await liveVideoRef.current.play();
      }
    } catch (err) {
      console.error(err);
      alert("Cannot access camera. Check permissions.");
    }
  }

  function stopCamera() {
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
    if (liveVideoRef.current) liveVideoRef.current.srcObject = null;
  }

  function startRecording() {
    if (!stream) {
      alert("Please start camera first.");
      return;
    }

    chunksRef.current = [];
    setRecordedBlob(null);
    setSavedName("");

    const options: MediaRecorderOptions = (() => {
      const candidates = [
        "video/webm;codecs=vp9",
        "video/webm;codecs=vp8",
        "video/webm",
      ];
      for (const mimeType of candidates) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          return { mimeType };
        }
      }
      return {};
    })();

    const recorder = new MediaRecorder(stream, options);
    recorderRef.current = recorder;

    recorder.ondataavailable = (e: BlobEvent) => {
      if (e.data && e.data.size > 0) {
        chunksRef.current.push(e.data);
      }
    };

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, {
        type: recorder.mimeType || "video/webm",
      });
      setRecordedBlob(blob);
    };

    recorder.start();
    setRecording(true);
  }

  function stopRecording() {
    recorderRef.current?.stop();
    setRecording(false);
  }

  async function uploadVideo() {
    if (!recordedBlob) {
      alert("No video recorded.");
      return;
    }

    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", recordedBlob, "tennis.webm");

      const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }

      const data = (await response.json()) as UploadResponse;
      setSavedName(data.saved);
      alert("Upload successful!");
    } catch (err: any) {
      console.error(err);
      alert(err?.message ?? "Upload error");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 16 }}>
      <h1>🎾 Tennis AI Coach Demo</h1>

      <video
        ref={liveVideoRef}
        playsInline
        muted
        style={{ width: "100%", borderRadius: 12, background: "#000" }}
      />

      <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
        <button onClick={startCamera} disabled={!!stream}>
          Start Camera
        </button>

        <button onClick={stopCamera} disabled={!stream}>
          Stop Camera
        </button>

        <button onClick={startRecording} disabled={!stream || recording}>
          Start Recording
        </button>

        <button onClick={stopRecording} disabled={!recording}>
          Stop Recording
        </button>

        <button onClick={uploadVideo} disabled={!recordedBlob || uploading}>
          {uploading ? "Uploading..." : "Upload Video"}
        </button>
      </div>

      {recordedBlob && (
        <div style={{ marginTop: 16 }}>
          <h3>Playback Preview</h3>
          <video
            controls
            style={{ width: "100%", borderRadius: 12 }}
            src={URL.createObjectURL(recordedBlob)}
          />
        </div>
      )}

      {savedName && (
        <p style={{ marginTop: 12 }}>
          Saved on backend as: <b>{savedName}</b>
        </p>
      )}
    </div>
  );
}
