import streamlit as st
import requests
import cv2
import tempfile
import os
import json
import base64
import gc

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
MAX_UPLOAD_MB = 30  # a little under Cloud Run's 32MB request-body cap, as a safety margin
MEDIA_WIDTH = 340    # shared fixed width for the preview image and the result video, side by side

st.set_page_config(page_title="VERITAS AI | Tactical Redaction System", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Main Body & Background Customization */
html, body, [class*="css"] { 
    font-family: 'Plus Jakarta Sans', sans-serif; 
}

[data-testid="stAppViewContainer"], [data-testid="stHeader"] { 
    background-color: #070A0F; 
}

body, p, span, label, [data-testid="stMarkdownContainer"] { 
    color: #CBD5E1; 
}

/* Dynamic Green-Cyan Gradient Heading Banner */
.gov-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 24px 30px;
    background: linear-gradient(135deg, #059669 0%, #0D9488 40%, #06B6D4 100%);
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -5px rgba(6, 182, 212, 0.25), 0 0 15px rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.gov-kicker {
    color: #A7F3D0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 4px;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

.gov-header h1 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800 !important;
    font-size: 2.1rem !important;
    color: #FFFFFF !important;
    margin: 0 !important;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.gov-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.2);
    color: #FFFFFF;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    margin-left: 12px;
    vertical-align: middle;
}

/* System Status Pill */
.status-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 8px 18px;
    border-radius: 30px;
    letter-spacing: 0.08em;
    white-space: nowrap;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

.status-ok { 
    background: #022C22; 
    color: #34D399; 
    border: 1.5px solid #10B981; 
}

.status-error { 
    background: #450A0A; 
    color: #FCA5A5; 
    border: 1.5px solid #EF4444; 
}

/* Instruction Briefing Drawer Styling */
.instruction-card {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-left: 5px solid #06B6D4;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 20px;
}

.instruction-card h4 {
    color: #22D3EE;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.instruction-card p, .instruction-card li {
    font-size: 0.90rem;
    color: #94A3B8;
    line-height: 1.5;
}

/* Step Containers with Glassmorphism Glow */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #0D131F;
    border: 1px solid #1E293B !important;
    border-radius: 10px;
    transition: all 0.3s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #0D9488 !important;
    box-shadow: 0 0 20px rgba(13, 148, 136, 0.15);
}

.step-label {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    color: #06B6D4;
    background: rgba(6, 182, 212, 0.1);
    border: 1px solid rgba(6, 182, 212, 0.25);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}

.step-heading {
    color: #F8FAFC;
    font-weight: 700;
    font-size: 1.25rem;
    margin-bottom: 4px;
}

.step-help {
    color: #64748B;
    font-size: 0.88rem;
    margin-bottom: 18px;
}

.empty-state {
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    padding: 40px 0;
    text-align: center;
    border: 1px dashed #1E293B;
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.4);
}

/* Custom Interactive Buttons */
.stButton button[kind="primary"], .stDownloadButton button {
    border: none !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.6rem 0 !important;
    border-radius: 6px !important;
    transition: all 0.2s ease-in-out !important;
}

.st-key-btn_preview button { 
    background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important; 
    color: #FFFFFF !important; 
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
}

.st-key-btn_preview button:hover { 
    background: linear-gradient(135deg, #38BDF8 0%, #0EA5E9 100%) !important; 
    box-shadow: 0 6px 18px rgba(56, 189, 248, 0.45);
}

.st-key-btn_process button { 
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important; 
    color: #FFFFFF !important; 
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}

.st-key-btn_process button:hover { 
    background: linear-gradient(135deg, #34D399 0%, #10B981 100%) !important; 
    box-shadow: 0 6px 18px rgba(52, 211, 153, 0.45);
}

.st-key-btn_download button { 
    background: linear-gradient(135deg, #06B6D4 0%, #0891B2 100%) !important; 
    color: #FFFFFF !important; 
    box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
}

.st-key-btn_download button:hover { 
    background: linear-gradient(135deg, #22D3EE 0%, #06B6D4 100%) !important; 
    box-shadow: 0 6px 18px rgba(34, 211, 238, 0.45);
}

/* File Upload Highlight */
[data-testid="stFileUploader"] button {
    background: #1E293B !important;
    color: #22D3EE !important;
    border: 1px solid #0891B2 !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #0891B2 !important;
    color: #FFFFFF !important;
}

/* Technical Badges & Sliders */
.tech-readout {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #22D3EE;
    background: rgba(6, 182, 212, 0.1);
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 6px;
    padding: 5px 12px;
    display: inline-block;
    margin: 8px 0 14px 0;
}

[data-testid="stSlider"] [role="slider"] { 
    background-color: #06B6D4 !important; 
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- state

for key, default in [
    ("job_id", None),
    ("video_signature", None),   # (name, size) - detects a new upload so we can reset job_id
    ("preview_image", None),
    ("preview_second", 0.0),
    ("result_video", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def reset_job_state():
    st.session_state.job_id = None
    st.session_state.preview_image = None
    st.session_state.result_video = None
    # Force immediate RAM recovery
    gc.collect()

def get_video_duration(file_bytes: bytes) -> float:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    cap.release()
    os.remove(tmp_path)
    return frame_count / fps if fps else 0


def check_backend():
    """Returns the /health response dict if the API is reachable AND gave a sane answer, otherwise None."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        body = r.json()
    except (requests.exceptions.RequestException, ValueError):
        return None
    return body if "pipeline_ready" in body else None


def _error_detail(r: requests.Response) -> str:
    try:
        return r.json().get("detail", r.text)
    except ValueError:
        return r.text


def call_preview(video_file, second: float, classes: str, progress_slot):
    data = {"second": str(second), "classes": classes}
    files = None
    if st.session_state.job_id:
        data["job_id"] = st.session_state.job_id
    else:
        files = {"video": (video_file.name, video_file.getvalue(), "video/mp4")}
    try:
        with requests.post(f"{API_URL}/preview", files=files, data=data, timeout=120, stream=True) as r:
            if r.status_code != 200:
                return None, _error_detail(r)
            if "X-Job-Id" in r.headers:
                st.session_state.job_id = r.headers["X-Job-Id"]
            image_bytes, err = None, None
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                msg = json.loads(line)
                if msg["type"] == "progress":
                    frac = msg["current"] / max(1, msg["total"])
                    progress_slot.progress(min(1.0, frac), text=f"Analyzing frame {msg['current']}/{msg['total']}")
                elif msg["type"] == "result":
                    image_bytes = base64.b64decode(msg["image_b64"])
                elif msg["type"] == "error":
                    err = msg["detail"]
            return image_bytes, err
    except requests.exceptions.RequestException as e:
        return None, str(e)


def call_process(effect, scope, selected_ids, classes,
                  max_seconds, sam_every_n_frames, show_trail, progress_slot):
    data = {
        "job_id": st.session_state.job_id,
        "effect": effect, 
        "scope": scope, 
        "selected_ids": selected_ids, 
        "classes": classes,
        "sam_every_n_frames": str(sam_every_n_frames), 
        "show_trail": str(show_trail).lower(),
    }
    if max_seconds is not None:
        data["max_seconds"] = str(max_seconds)
        
    try:
        with requests.post(f"{API_URL}/process", data=data, timeout=600, stream=True) as r:
            if r.status_code != 200:
                return None, None, _error_detail(r)
            
            video_bytes, metrics, err = None, None, None
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                msg = json.loads(line)
                if msg["type"] == "progress":
                    frac = msg["current"] / max(1, msg["total"])
                    progress_slot.progress(min(1.0, frac), text=f"Redacting frame {msg['current']}/{msg['total']}")
                elif msg["type"] == "result":
                    video_bytes = base64.b64decode(msg["video_b64"])
                    metrics = msg.get("metrics")
                elif msg["type"] == "error":
                    err = msg["detail"]
                    
            return video_bytes, metrics, err

    except requests.exceptions.Timeout:
        return None, None, "Request timed out - try a shorter clip or larger sam_every_n_frames."
    except requests.exceptions.RequestException as e:
        return None, None, str(e)


# --------------------------------------------------------------- header

health = check_backend()
status_html = (
    '<div class="status-pill status-ok">● SYSTEM OPERATIONAL</div>' if (health and health["pipeline_ready"])
    else '<div class="status-pill status-error">● PIPELINE OFFLINE</div>'
)

st.markdown(f"""
<div class="gov-header">
  <div>
    <div class="gov-kicker">GOV-TECH COMPLIANCE & PRIVACY SUITE</div>
    <h1>VERITAS AI <span class="gov-badge">PRO v2.4</span></h1>
  </div>
  {status_html}
</div>
""", unsafe_allow_html=True)

if health is None:
    st.error("API unreachable. Start the backend with: `uvicorn api:app --reload --port 8000`")
    st.stop()
elif not health["pipeline_ready"]:
    st.error(f"Engine offline: {health['error']}")
    st.stop()


# --------------------------------------------------------- about & how-to (collapsible)

with st.expander("📖 Information & Operational Guide", expanded=False):
    st.markdown("""
    <div class="instruction-card">
    <h4>VERITAS Target-Specific Video Redaction Architecture</h4>
    <p>Designed for public records, evidentiary compliance, and law enforcement privacy masking. 
    It tracks targets across frames via YOLO11 + ByteTrack and applies precision MobileSAM mask segmentation.</p>
    <ol>
        <li><b>Ingest & Target (Phase 1):</b> Ingest raw video and select a keyframe timestamp to generate track ID boxes.</li>
        <li><b>Configure Scope (Phase 2):</b> Identify targets by ID and set redaction mode (Blur or Pixelate).</li>
        <li><b>Process & Export:</b> Run SAM-assisted temporal masking and download compliant footage.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------- two-column workspace

col_preview, col_process = st.columns(2, gap="large", border=False)

with col_preview:
    step1 = st.container(border=True)
    with step1:
        st.markdown('<div class="step-label">PHASE 1</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-heading">Ingestion & Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-help">Analyze timestamp keyframe to assign unique Target IDs.</div>', unsafe_allow_html=True)

        video_file = st.file_uploader("Video file", type=["mp4", "avi", "mov"], label_visibility="collapsed")

        duration = None
        if video_file is not None:
            signature = (video_file.name, video_file.size)
            if signature != st.session_state.video_signature:
                st.session_state.video_signature = signature
                reset_job_state()

            size_mb = video_file.size / (1024 * 1024)
            if size_mb > MAX_UPLOAD_MB:
                st.error(f"File size {size_mb:.1f}MB exceeds limit ({MAX_UPLOAD_MB}MB). Trim footage prior to upload.")
                st.stop()

            duration = get_video_duration(video_file.getvalue())
            st.markdown(f'<span class="tech-readout">PAYLOAD: {duration:.1f}s | {size_mb:.1f} MB</span>', unsafe_allow_html=True)

            second = st.slider("Preview Keyframe (seconds)", min_value=0.0,
                                max_value=max(0.1, round(duration, 1)),
                                value=min(st.session_state.preview_second, duration), step=0.5)
            st.session_state.preview_second = second

            classes = st.text_input("COCO Target Classes", value="0",
                                     help="Class IDs, comma-separated. Default: 0 (Person).")

            if st.button("Generate Frame Analysis", key="btn_preview", type="primary", use_container_width=True):
                progress_slot = st.empty()
                image_bytes, err = call_preview(video_file, second, classes, progress_slot)
                progress_slot.empty()
                if err:
                    st.error(f"Detection failed: {err}")
                else:
                    st.session_state.preview_image = image_bytes
        else:
            classes = "0"

        if st.session_state.preview_image is not None:
            st.image(st.session_state.preview_image, width=MEDIA_WIDTH,
                      caption="Target IDs resolved. Note IDs for target targeting.")
        else:
            st.markdown('<div class="empty-state">AWAITING FOOTAGE INGESTION</div>', unsafe_allow_html=True)

with col_process:
    step2 = st.container(border=True)
    with step2:
        st.markdown('<div class="step-label">PHASE 2</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-heading">Redaction Engine</div>', unsafe_allow_html=True)

        if not (st.session_state.job_id and st.session_state.preview_image is not None):
            st.markdown('<div class="step-help">Complete Phase 1 detection to activate engine controls.</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="empty-state">AWAITING PHASE 1 ANALYSIS</div>', unsafe_allow_html=True)
        else:
            effect = st.radio("Masking Mode", options=["pixelate", "blur"], horizontal=True)
            scope = st.radio(
                "Redaction Scope", options=["selected", "background"], horizontal=True,
                help="selected: redact specified IDs. background: protect specified IDs, redact all others.",
            )
            selected_ids = st.text_input("Target Track ID(s)", placeholder="e.g. 1, 3, 4",
                                          help="Comma-separated IDs identified in Phase 1.")

            with st.expander("Advanced Engine Parameters"):
                max_seconds = st.slider("Processing Duration (s)", min_value=1.0,
                                         max_value=max(1.0, round(duration, 1)) if duration else 30.0,
                                         value=min(15.0, duration) if duration else 15.0, step=1.0,
                                         help="Limits overall execution length.")
                sam_every_n_frames = st.number_input(
                    "MobileSAM Refinement Interval", min_value=1, max_value=30, value=5,
                    help="Cadence of SAM mask segmentation updates.",
                )
                show_trail = st.checkbox("Overlay Motion Vectors", value=False,
                                          help="Visualizes movement paths.")

            show_metrics = st.checkbox("Display Execution Analytics", value=False, help="Show FPS and rendering performance stats.")

            process_clicked = st.button(
                "Execute Redaction Pipeline", key="btn_process", type="primary", use_container_width=True,
                disabled=(scope == "selected" and not selected_ids.strip()),
            )

            if process_clicked:
                progress_slot = st.empty()
                video_bytes, metrics, err = call_process(effect, scope, selected_ids, classes,
                                                          max_seconds, sam_every_n_frames, show_trail, progress_slot)
                progress_slot.empty()
                if err:
                    st.error(f"Processing failed: {err}")
                else:
                    st.session_state.result_video = video_bytes
                    st.session_state.last_metrics = metrics
                    st.success("Execution Completed Successfully.")

            if st.session_state.result_video is not None:
                st.video(st.session_state.result_video, width=MEDIA_WIDTH)
                
                if show_metrics and st.session_state.get("last_metrics"):
                    m = st.session_state.last_metrics
                    with st.expander("📊 Performance Analytics", expanded=True):
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Processing Speed", f"{m.get('fps', 0)} FPS")
                        m2.metric("Total Elapsed", f"{m.get('elapsed_seconds', 0)}s")
                        m3.metric("Processed Frames", f"{m.get('total_frames', 0)}")
                        
                        st.caption(f"**MobileSAM Mask Refinements:** {m.get('sam_refine_calls', 0)} calls across {m.get('active_ids', 0)} active track ID(s).")

                st.download_button("Export Redacted Video", key="btn_download",
                                    data=st.session_state.result_video,
                                    file_name="redacted_export.mp4", mime="video/mp4", use_container_width=True)