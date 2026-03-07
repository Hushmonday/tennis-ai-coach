import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

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
  perfect_standard?: {
    focus?: string;
    targets?: string[];
  };
  coach_feedback?: unknown;
  nova_error?: unknown;
};

type CoachIssue = {
  name: string;
  why: string;
  fixes: string[];
};

type CoachView = {
  encouragement: string;
  closing: string;
  issues: CoachIssue[];
  perfectFocus: string;
  perfectTargets: string[];
};

type InputMode = "record" | "upload";
type Handedness = "right" | "left";
type ShotType = "forehand" | "backhand" | "serve";

function asDisplayText(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function normalizeLabel(raw: string): string {
  return raw
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function parseCoachFeedback(
  raw: unknown,
  fallbackPerfect?: { focus?: string; targets?: string[] },
): CoachView {
  const fallback: CoachView = {
    encouragement: "Nice effort. Your swing shows a good base to build on.",
    closing: "Keep training with consistent camera angle and lighting for better feedback.",
    issues: [],
    perfectFocus: fallbackPerfect?.focus || "Shot Standard",
    perfectTargets: fallbackPerfect?.targets || [],
  };

  if (!raw || typeof raw !== "object") return fallback;

  const data = raw as Record<string, unknown>;
  const encouragement =
    typeof data.encouragement === "string" && data.encouragement.trim().length > 0
      ? data.encouragement
      : fallback.encouragement;
  const closing =
    typeof data.closing === "string" && data.closing.trim().length > 0 ? data.closing : fallback.closing;

  const issuesRaw = data.issues;
  const issues: CoachIssue[] = [];
  if (issuesRaw && typeof issuesRaw === "object") {
    for (const [key, value] of Object.entries(issuesRaw as Record<string, unknown>)) {
      if (!value || typeof value !== "object") continue;
      const v = value as Record<string, unknown>;
      const why = typeof v.why_it_matters === "string" ? v.why_it_matters : "";
      const fixes = Array.isArray(v.fixes) ? v.fixes.filter((x): x is string => typeof x === "string") : [];
      issues.push({
        name: normalizeLabel(key),
        why,
        fixes,
      });
    }
  }

  const perfectRaw = data.perfect_standard;
  let perfectFocus = fallback.perfectFocus;
  let perfectTargets = fallback.perfectTargets;
  if (perfectRaw && typeof perfectRaw === "object") {
    const p = perfectRaw as Record<string, unknown>;
    if (typeof p.focus === "string" && p.focus.trim().length > 0) perfectFocus = p.focus;
    if (Array.isArray(p.targets)) {
      perfectTargets = p.targets.filter((x): x is string => typeof x === "string");
    }
  }

  return {
    encouragement,
    closing,
    issues,
    perfectFocus,
    perfectTargets,
  };
}

function formatMetricValue(value: number): string {
  if (!Number.isFinite(value)) return "-";
  return Math.abs(value) >= 100 ? value.toFixed(1) : value.toFixed(3);
}

function metricExplanation(key: string): { title: string; meaning: string } {
  const m: Record<string, { title: string; meaning: string }> = {
    elbow_angle_at_contact_deg: {
      title: "Elbow Angle At Contact (deg)",
      meaning: "Angle at your elbow when the ball is contacted. Bigger number = straighter arm.",
    },
    wrist_vs_shoulder_normY: {
      title: "Wrist vs Shoulder Height",
      meaning: "Relative vertical wrist position vs shoulder. Negative = wrist higher, positive = wrist lower.",
    },
    wrist_vs_hip_normY: {
      title: "Wrist vs Hip Height",
      meaning: "Relative vertical wrist position vs hip center. Negative = wrist above hips, positive = below hips.",
    },
    wrist_minus_hipcenter_normX: {
      title: "Contact In Front Of Body",
      meaning: "Horizontal distance from wrist to hip center at contact. Larger absolute value = farther from body.",
    },
    follow_through_disp_norm: {
      title: "Follow-through Displacement",
      meaning: "How far the wrist travels after contact (normalized by frame width). Bigger = stronger follow-through.",
    },
    wrist_to_hipcenter_norm: {
      title: "Wrist To Hip Center Distance",
      meaning: "Distance from wrist to torso center at contact. Too small may mean contact is too close to body.",
    },
    min_knee_angle_deg: {
      title: "Minimum Knee Angle (deg)",
      meaning: "Smallest knee angle during motion. Smaller = deeper knee bend.",
    },
    wrist_vs_shoulder_normY_at_contact: {
      title: "Serve Contact Height",
      meaning: "Vertical wrist position vs shoulder at serve contact. Negative usually means higher contact.",
    },
    hip_upward_drive_norm: {
      title: "Upward Drive",
      meaning: "How much hips move upward into contact during serve. Bigger = stronger leg drive.",
    },
  };
  return m[key] || {
    title: normalizeLabel(key),
    meaning: "Model-generated metric for swing analysis.",
  };
}

export default function App() {
  const liveVideoRef = useRef<HTMLVideoElement | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  const [mode, setMode] = useState<InputMode>("record");

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [recording, setRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [recordedPreviewUrl, setRecordedPreviewUrl] = useState<string | null>(null);

  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedPreviewUrl, setUploadedPreviewUrl] = useState<string | null>(null);

  const [handedness, setHandedness] = useState<Handedness>("right");
  const [shotType, setShotType] = useState<ShotType>("forehand");

  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  const coachView = useMemo(
    () => parseCoachFeedback(result?.coach_feedback, result?.perfect_standard),
    [result?.coach_feedback, result?.perfect_standard],
  );

  useEffect(() => {
    return () => stopCamera();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    return () => {
      if (recordedPreviewUrl) URL.revokeObjectURL(recordedPreviewUrl);
    };
  }, [recordedPreviewUrl]);

  useEffect(() => {
    return () => {
      if (uploadedPreviewUrl) URL.revokeObjectURL(uploadedPreviewUrl);
    };
  }, [uploadedPreviewUrl]);

  async function startCamera() {
    if (stream) return;
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
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
    setRecording(false);
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
    if (liveVideoRef.current) liveVideoRef.current.srcObject = null;
  }

  function startRecording() {
    if (!stream) return alert("Start camera first.");

    chunksRef.current = [];
    setRecordedBlob(null);
    setResult(null);
    if (recordedPreviewUrl) {
      URL.revokeObjectURL(recordedPreviewUrl);
      setRecordedPreviewUrl(null);
    }

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
      setRecordedPreviewUrl(URL.createObjectURL(blob));
    };

    recorder.start();
    setRecording(true);
  }

  function stopRecording() {
    recorderRef.current?.stop();
    setRecording(false);
  }

  function onFilePicked(file: File | null) {
    if (!file) return;

    if (!file.type.startsWith("video/")) {
      alert("Please choose a video file.");
      return;
    }

    setResult(null);
    setUploadedFile(file);
    if (uploadedPreviewUrl) URL.revokeObjectURL(uploadedPreviewUrl);
    setUploadedPreviewUrl(URL.createObjectURL(file));
  }

  function getAnalyzeTarget() {
    if (mode === "upload" && uploadedFile) {
      return {
        file: uploadedFile,
        fileName: uploadedFile.name,
      };
    }

    if (mode === "record" && recordedBlob) {
      return {
        file: recordedBlob,
        fileName: "tennis.webm",
      };
    }

    return null;
  }

  async function analyze() {
    const target = getAnalyzeTarget();
    if (!target) {
      alert(mode === "upload" ? "Upload a video first." : "Record a video first.");
      return;
    }

    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", target.file, target.fileName);
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
    <div className="page">
      <div className="bg-glow bg-glow-a" />
      <div className="bg-glow bg-glow-b" />

      <main className="app-shell">
        <section className="hero">
          <p className="eyebrow">Swing Intelligence</p>
          <h1>Tennis AI Coach</h1>
          <p className="subtitle">
            Upload a swing video or record directly in the browser. For best accuracy, use a clean side
            view and keep the full body in frame.
          </p>
        </section>

        <section className="controls-panel">
          <div className="mode-switch" role="tablist" aria-label="Input mode">
            <button
              className={mode === "record" ? "mode-btn active" : "mode-btn"}
              onClick={() => setMode("record")}
              type="button"
            >
              Record
            </button>
            <button
              className={mode === "upload" ? "mode-btn active" : "mode-btn"}
              onClick={() => setMode("upload")}
              type="button"
            >
              Upload
            </button>
          </div>

          <div className="selectors">
            <label>
              Handedness
              <select value={handedness} onChange={(e) => setHandedness(e.target.value as Handedness)}>
                <option value="right">Right-handed</option>
                <option value="left">Left-handed</option>
              </select>
            </label>

            <label>
              Shot Type
              <select value={shotType} onChange={(e) => setShotType(e.target.value as ShotType)}>
                <option value="forehand">Forehand</option>
                <option value="backhand">Backhand</option>
                <option value="serve">Serve</option>
              </select>
            </label>
          </div>
        </section>

        <section className="video-panel">
          {mode === "record" ? (
            <>
              <video ref={liveVideoRef} playsInline muted className="live-video" />
              <div className="button-row">
                <button type="button" onClick={startCamera} disabled={!!stream}>
                  Start Camera
                </button>
                <button type="button" onClick={stopCamera} disabled={!stream}>
                  Stop Camera
                </button>
                <button type="button" onClick={startRecording} disabled={!stream || recording}>
                  Start Recording
                </button>
                <button type="button" onClick={stopRecording} disabled={!recording}>
                  Stop Recording
                </button>
              </div>
            </>
          ) : (
            <>
              <label className="upload-zone" htmlFor="video-upload">
                <input
                  id="video-upload"
                  type="file"
                  accept="video/*"
                  onChange={(e) => onFilePicked(e.target.files?.[0] ?? null)}
                />
                <span>{uploadedFile ? uploadedFile.name : "Choose a video file"}</span>
                <small>MP4, MOV, WEBM and other video formats are accepted</small>
              </label>
            </>
          )}

          <button className="analyze-btn" type="button" onClick={analyze} disabled={uploading}>
            {uploading ? "Analyzing..." : "Upload & Analyze"}
          </button>

          {recordedPreviewUrl && mode === "record" && (
            <div className="preview-block">
              <h3>Recorded Preview</h3>
              <video controls className="preview-video" src={recordedPreviewUrl} />
            </div>
          )}

          {uploadedPreviewUrl && mode === "upload" && (
            <div className="preview-block">
              <h3>Uploaded Preview</h3>
              <video controls className="preview-video" src={uploadedPreviewUrl} />
            </div>
          )}
        </section>

        {result && (
          <section className="result-panel">
            <h2>Analysis Result</h2>

            {!result.ok ? (
              <div className="result-card bad">
                <h3>Tracking quality is too low</h3>
                <p>{result.message}</p>
              </div>
            ) : (
              <>
                <div className="result-grid">
                  <article className="result-card">
                    <h3>Overview</h3>
                    <p>
                      <strong>Quality:</strong> {result.quality_score?.toFixed(2)}
                    </p>
                    <p>
                      <strong>Camera:</strong> {result.camera_angle}
                    </p>
                    <p>
                      <strong>Contact Frame:</strong> {result.contact_frame_index}
                    </p>
                    {result.warning && (
                      <p className="warn">
                        <strong>Warning:</strong> {result.warning}
                      </p>
                    )}
                  </article>

                  <article className="result-card">
                    <h3>Coaching Flags</h3>
                    {result.flags && result.flags.length > 0 ? (
                      <div className="chip-list">
                        {result.flags.map((flag) => (
                          <span className="chip" key={flag}>
                            {normalizeLabel(flag)}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <p>No major flags.</p>
                    )}
                  </article>

                  <article className="result-card">
                    <h3>Metrics</h3>
                    <ul className="metric-list">
                      {Object.entries(result.metrics || {}).map(([key, value]) => (
                        <li key={key}>
                          <div className="metric-main">
                            <span>{metricExplanation(key).title}</span>
                            <small>{metricExplanation(key).meaning}</small>
                          </div>
                          <strong>{formatMetricValue(value)}</strong>
                        </li>
                      ))}
                    </ul>
                  </article>
                </div>

                <article className="result-card coach-text">
                  <h3>Coach Feedback</h3>
                  <p className="coach-lead">{coachView.encouragement}</p>

                  {coachView.issues.length > 0 ? (
                    <div className="issue-list">
                      {coachView.issues.map((issue) => (
                        <div key={issue.name} className="issue-item">
                          <h4>{issue.name}</h4>
                          {issue.why && <p>{issue.why}</p>}
                          {issue.fixes.length > 0 && (
                            <ul>
                              {issue.fixes.map((fix) => (
                                <li key={fix}>{fix}</li>
                              ))}
                            </ul>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p>No critical issue detected from this clip.</p>
                  )}

                  {coachView.perfectTargets.length > 0 && (
                    <>
                      <h4>{coachView.perfectFocus}</h4>
                      <ul>
                        {coachView.perfectTargets.map((target) => (
                          <li key={target}>{target}</li>
                        ))}
                      </ul>
                    </>
                  )}

                  <p>{coachView.closing}</p>
                </article>

                {result.nova_error && (
                  <article className="result-card bad">
                    <h3>Nova Error</h3>
                    <pre>{asDisplayText(result.nova_error)}</pre>
                  </article>
                )}
              </>
            )}
          </section>
        )}
      </main>
    </div>
  );
}
