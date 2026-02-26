import { useEffect, useRef, useState } from "react";

type AnalyzeResponse = {
  ok: boolean;
  saved?: string;
  message?: string;
  warning?: string | null;
  quality_score?: number;
  camera_angle?: string;
  contact_frame_index?: number;
  metrics?: Record<string, number>;
  flags?: string[];
  perfect_standard?: any;
};

export default function App() {
  const liveVideoRef = useRef<HTMLVideoElement | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [recording, setRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);

  const [handedness, setHandedness] = useState<"right" | "left">("right");
  const [shotType, setShotType] = useState<"forehand" | "backhand" | "serve">("forehand");

  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  useEffect(() => {
    void startCamera();
    return () => stopCamera();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function startCamera() {
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, frameRate: 30 },
        audio: false,
      });
      setStream(s);
      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = s;
        await liveVideoRef.current.play();
      }
    } catch (e) {
      console.error(e);
      alert("Cannot access camera. Check browser permissions.");
    }
  }

  function stopCamera() {
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
    if (liveVideoRef.current) liveVideoRef.current.srcObject = null;
  }

  function startRecording() {
    if (!stream) return alert("Start camera first.");
    chunksRef.current = [];
    setRecordedBlob(null);
    setResult(null);

    const options: MediaRecorderOptions = (() => {
      const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
      for (const mimeType of candidates) {
        if (MediaRecorder.isTypeSupported(mimeType)) return { mimeType };
      }
      return {};
    })();

    const recorder = new MediaRecorder(stream, options);
    recorderRef.current = recorder;

    recorder.ondataavailable = (e: BlobEvent) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "video/webm" });
      setRecordedBlob(blob);
    };

    recorder.start();
    setRecording(true);
  }

  function stopRecording() {
    recorderRef.current?.stop();
    setRecording(false);
  }

  async function analyze() {
    if (!recordedBlob) return alert("Record a video first.");
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", recordedBlob, "tennis.webm");
      fd.append("handedness", handedness);
      fd.append("shot_type", shotType);

      const resp = await fetch("/api/analyze", {
        method: "POST",
        body: fd,
      });
      const data = (await resp.json()) as AnalyzeResponse;
      setResult(data);
    } catch (e) {
      console.error(e);
      alert("Analyze request failed.");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: 16 }}>
      <h1>🎾 Tennis AI Coach Demo</h1>

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 12 }}>
        <label>
          Handedness:&nbsp;
          <select value={handedness} onChange={(e) => setHandedness(e.target.value as any)}>
            <option value="right">Right-handed</option>
            <option value="left">Left-handed</option>
          </select>
        </label>

        <label>
          Shot type:&nbsp;
          <select value={shotType} onChange={(e) => setShotType(e.target.value as any)}>
            <option value="forehand">Forehand</option>
            <option value="backhand">Backhand</option>
            <option value="serve">Serve</option>
          </select>
        </label>

        <div style={{ color: "#555" }}>
          Recording tip: <b>side view (90°)</b>, full body in frame.
        </div>
      </div>

      <video
        ref={liveVideoRef}
        playsInline
        muted
        style={{ width: "100%", borderRadius: 12, background: "#000" }}
      />

      <div style={{ display: "flex", gap: 12, marginTop: 12, flexWrap: "wrap" }}>
        <button onClick={startRecording} disabled={!stream || recording}>
          Start Recording
        </button>
        <button onClick={stopRecording} disabled={!recording}>
          Stop Recording
        </button>
        <button onClick={analyze} disabled={!recordedBlob || uploading}>
          {uploading ? "Analyzing..." : "Upload & Analyze"}
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

      {result && (
        <div style={{ marginTop: 16 }}>
          <h3>Analysis Result</h3>

          {!result.ok && (
            <div style={{ padding: 12, borderRadius: 12, background: "#ffecec" }}>
              <b>Not enough tracking quality.</b>
              <div>{result.message}</div>
            </div>
          )}

          {result.ok && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div style={{ padding: 12, borderRadius: 12, background: "#f6f6f6" }}>
                <div><b>Quality:</b> {result.quality_score?.toFixed(2)}</div>
                <div><b>Camera angle:</b> {result.camera_angle}</div>
                {result.warning && <div style={{ color: "#a15c00" }}><b>Warning:</b> {result.warning}</div>}
                <div><b>Contact frame idx:</b> {result.contact_frame_index}</div>
              </div>

              <div style={{ padding: 12, borderRadius: 12, background: "#f6f6f6" }}>
                <div><b>Flags:</b> {(result.flags || []).join(", ") || "None"}</div>
                <div style={{ marginTop: 8 }}><b>Perfect standard:</b></div>
                <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>
                  {JSON.stringify(result.perfect_standard, null, 2)}
                </pre>
              </div>

              <div style={{ gridColumn: "1 / span 2", padding: 12, borderRadius: 12, background: "#111", color: "#0f0" }}>
                <b>Metrics</b>
                <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>
                  {JSON.stringify(result.metrics, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}