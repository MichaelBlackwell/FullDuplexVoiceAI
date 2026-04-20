let pc = null;
let localStream = null;
let peerId = null;

const statusEl = document.getElementById("status");
const latencyEl = document.getElementById("latency");
const btnConnect = document.getElementById("btn-connect");
const btnDisconnect = document.getElementById("btn-disconnect");

function setStatus(state, text) {
    statusEl.textContent = text || state;
    statusEl.className = "status " + state;
}

async function connect() {
    btnConnect.disabled = true;
    setStatus("connecting", "Connecting...");

    try {
        // Capture microphone
        localStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });

        // Create peer connection — no ICE servers needed for localhost
        pc = new RTCPeerConnection();

        // Handle remote track (audio from server)
        pc.ontrack = (event) => {
            const audioEl = document.getElementById("remote-audio");
            audioEl.srcObject = event.streams[0];
        };

        // Add local mic track
        localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));

        // Create and set local offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // Wait for ICE gathering to complete
        await new Promise((resolve) => {
            if (pc.iceGatheringState === "complete") {
                resolve();
            } else {
                const check = () => {
                    if (pc.iceGatheringState === "complete") {
                        pc.removeEventListener("icegatheringstatechange", check);
                        resolve();
                    }
                };
                pc.addEventListener("icegatheringstatechange", check);
            }
        });

        // Send offer to server, get answer
        const response = await fetch("/offer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sdp: pc.localDescription.sdp,
                type: pc.localDescription.type,
            }),
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const answer = await response.json();
        peerId = answer.peer_id;
        await pc.setRemoteDescription(answer);

        // Connect WebSocket for transcripts
        connectTranscriptWs();

        // Monitor connection state
        pc.onconnectionstatechange = () => {
            const state = pc.connectionState;
            switch (state) {
                case "connected":
                    setStatus("connected", "Connected");
                    btnDisconnect.disabled = false;
                    startLatencyMeasurement();
                    break;
                case "disconnected":
                case "failed":
                    setStatus("failed", state === "failed" ? "Failed" : "Disconnected");
                    cleanup();
                    break;
                case "closed":
                    setStatus("disconnected", "Disconnected");
                    cleanup();
                    break;
            }
        };
    } catch (err) {
        console.error("Connection failed:", err);
        setStatus("failed", "Failed: " + err.message);
        cleanup();
    }
}

function disconnect() {
    cleanup();
    setStatus("disconnected", "Disconnected");
}

let transcriptWs = null;
let currentResponseEl = null;

function connectTranscriptWs() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    transcriptWs = new WebSocket(`${proto}//${location.host}/ws/transcripts`);
    transcriptWs.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            switch (data.type) {
                case "transcript":
                    appendTranscript(data.text);
                    currentResponseEl = startNewResponse();
                    break;
                case "llm_chunk":
                    appendResponseChunk(data.text);
                    break;
                case "llm_done":
                    finalizeResponse();
                    break;
                case "stop_playback":
                    handleStopPlayback();
                    break;
            }
        } catch (e) {
            console.error("Failed to parse message:", e);
        }
    };
}

function cleanup() {
    if (transcriptWs) {
        transcriptWs.close();
        transcriptWs = null;
    }
    if (localStream) {
        localStream.getTracks().forEach((track) => track.stop());
        localStream = null;
    }
    if (pc) {
        pc.close();
        pc = null;
    }
    peerId = null;
    btnConnect.disabled = false;
    btnDisconnect.disabled = true;
    latencyEl.textContent = "—";
}

function appendTranscript(text) {
    const container = document.getElementById("transcripts");
    const line = document.createElement("div");
    line.className = "transcript-line";

    const timeSpan = document.createElement("span");
    timeSpan.className = "transcript-time";
    timeSpan.textContent = new Date().toLocaleTimeString();

    line.appendChild(timeSpan);
    line.appendChild(document.createTextNode(text));
    container.appendChild(line);
    container.scrollTop = container.scrollHeight;
}

function startNewResponse() {
    const container = document.getElementById("ai-response");
    const line = document.createElement("div");
    line.className = "response-line";

    const timeSpan = document.createElement("span");
    timeSpan.className = "transcript-time";
    timeSpan.textContent = new Date().toLocaleTimeString();

    const textSpan = document.createElement("span");
    textSpan.className = "response-text";

    line.appendChild(timeSpan);
    line.appendChild(textSpan);
    container.appendChild(line);
    container.scrollTop = container.scrollHeight;
    return textSpan;
}

function appendResponseChunk(text) {
    if (!currentResponseEl) {
        currentResponseEl = startNewResponse();
    }
    currentResponseEl.textContent += text;
    const container = document.getElementById("ai-response");
    container.scrollTop = container.scrollHeight;
}

function finalizeResponse() {
    currentResponseEl = null;
}

function handleStopPlayback() {
    // Briefly mute to clear the browser's jitter buffer of stale audio
    const audioEl = document.getElementById("remote-audio");
    if (audioEl) {
        audioEl.muted = true;
        setTimeout(() => {
            audioEl.muted = false;
        }, 100);
    }
    // Mark current response as interrupted in the UI
    if (currentResponseEl) {
        currentResponseEl.textContent += " [interrupted]";
        currentResponseEl = null;
    }
}

// Simple latency estimation using the WebRTC stats API
let latencyInterval = null;

function startLatencyMeasurement() {
    if (latencyInterval) clearInterval(latencyInterval);
    latencyInterval = setInterval(async () => {
        if (!pc) {
            clearInterval(latencyInterval);
            return;
        }
        try {
            const stats = await pc.getStats();
            stats.forEach((report) => {
                if (report.type === "candidate-pair" && report.currentRoundTripTime !== undefined) {
                    const rttMs = (report.currentRoundTripTime * 1000).toFixed(0);
                    latencyEl.textContent = rttMs + " ms";
                }
            });
        } catch {
            // Stats not available yet
        }
    }, 1000);
}

// --- Settings Panel ---

function toggleSettings() {
    document.getElementById("settings-panel").classList.toggle("open");
}

async function loadDefaults() {
    try {
        const resp = await fetch("/settings/defaults");
        const data = await resp.json();
        const select = document.getElementById("voice-select");
        select.innerHTML = "";
        for (const v of data.voices) {
            const opt = document.createElement("option");
            opt.value = v;
            opt.textContent = v;
            if (v === data.default_voice) opt.selected = true;
            select.appendChild(opt);
        }
        document.getElementById("tts-instruct").value = data.default_instruct || "";
        document.getElementById("system-prompt").value = data.default_system_prompt || "";
    } catch (e) {
        console.error("Failed to load defaults:", e);
    }
}

async function applySettings() {
    if (!peerId) return;
    const btn = document.getElementById("btn-apply");
    try {
        const resp = await fetch("/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                peer_id: peerId,
                voice: document.getElementById("voice-select").value,
                instruct: document.getElementById("tts-instruct").value,
                system_prompt: document.getElementById("system-prompt").value,
            }),
        });
        if (resp.ok) {
            btn.classList.add("success");
            btn.textContent = "Applied";
            setTimeout(() => {
                btn.classList.remove("success");
                btn.textContent = "Apply Settings";
            }, 1500);
        }
    } catch (e) {
        console.error("Failed to apply settings:", e);
    }
}

// Load defaults on page load
loadDefaults();
