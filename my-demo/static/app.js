const $ = (id) => document.getElementById(id);      // short name for getElementById
let clip = null;                                     // the chosen video file
let picked = [];                                     // the picked frames: [{blob, time}]

// --- 1. Status pill: is the server reachable? ---
async function checkServer() {
  try {
    const data = await (await fetch("hello?name=phone")).json();
    $("status").className = "pill ok";
    $("status-text").textContent = `connected · ${data.gpus.length} GPUs`;
  } catch (error) {
    $("status").className = "pill err";
    $("status-text").textContent = "offline";
  }
}

// --- 2. A clip was recorded or chosen ---
function onClipChosen(event) {
  clip = event.target.files[0];
  if (!clip) return;
  $("picked").textContent = `${clip.name} · ${(clip.size / 1e6).toFixed(1)} MB`;
  $("preview").src = URL.createObjectURL(clip);
  $("preview").style.display = "block";
  $("sample").disabled = false;
}

// Wait until an element fires an event once (for example, the video finished jumping)
function waitFor(element, eventName) {
  return new Promise((resolve) => element.addEventListener(eventName, resolve, { once: true }));
}

// --- 3. Pick n evenly spaced frames from a video file, as JPEGs ---
async function pickFrames(file, n, maxSide) {
  const video = document.createElement("video");     // invisible video player
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = URL.createObjectURL(file);
  await waitFor(video, "loadeddata");                 // wait until the first picture is ready

  const duration = video.duration;
  const scale = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  const ctx = canvas.getContext("2d");

  const frames = [];
  for (let k = 0; k < n; k++) {
    const time = (k + 0.5) * duration / n;            // middle of part k (same rule as pick_indices)
    video.currentTime = time;                         // jump there...
    await waitFor(video, "seeked");                   // ...and wait until the jump is done
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
    frames.push({ blob, time });
  }
  URL.revokeObjectURL(video.src);                     // free the memory
  return { frames, duration, width: canvas.width, height: canvas.height };
}

// --- 4. "Pick frames" button: pick, then show them ---
async function onSample() {
  $("sample").disabled = true;
  const n = Number($("nframes").value);
  const side = Number($("side").value);
  try {
    const start = performance.now();
    const result = await pickFrames(clip, n, side);
    const seconds = (performance.now() - start) / 1000;
    picked = result.frames;

    const totalKB = picked.reduce((sum, f) => sum + f.blob.size, 0) / 1024;
    $("frames-info").textContent =
      `${n} frames · ${result.width}×${result.height} px · ${totalKB.toFixed(0)} KB in total · ` +
      `picked in ${seconds.toFixed(2)} s · clip ${result.duration.toFixed(1)} s`;

    $("frames").innerHTML = "";
    for (const f of picked) {
      const figure = document.createElement("figure");
      const img = document.createElement("img");
      img.src = URL.createObjectURL(f.blob);
      const caption = document.createElement("figcaption");
      caption.textContent = `${f.time.toFixed(2)} s`;
      figure.append(img, caption);
      $("frames").append(figure);
    }
  } catch (error) {
    $("frames-info").textContent = "Could not read this clip: " + error;
  }
  $("sample").disabled = false;
}

$("rec").addEventListener("change", onClipChosen);
$("pick").addEventListener("change", onClipChosen);
$("sample").addEventListener("click", onSample);
checkServer();

// --- 5. Send the picked frames, show the answer (Step 9.3) ---
async function onSend() {
  if (picked.length === 0) { $("result-info").textContent = "Pick frames first."; return; }
  $("send").disabled = true;
  $("answer").textContent = "";
  $("result-info").textContent = `Sending ${picked.length} frames and waiting for the model…`;

  const form = new FormData();                               // the request body: files + values
  picked.forEach((f, k) => form.append("frames", f.blob, `frame${k}.jpg`));
  const gap = picked.length > 1 ? picked[1].time - picked[0].time : 0;
  form.append("gap", gap.toFixed(3));

  const start = performance.now();
  try {
    const reply = await fetch("analyze_clip", { method: "POST", body: form });
    const data = await reply.json();
    const total = (performance.now() - start) / 1000;
    if (reply.status === 429) throw new Error("the GPUs are busy, try again in a few seconds");
    if (!reply.ok) throw new Error(data.error || `status ${reply.status}`);
    if (data.dropped) throw new Error("this clip was replaced by a newer segment");
    showResult(data, total);
  } catch (error) {
    $("result-info").textContent = "Error: " + error.message;
  }
  $("send").disabled = false;
}

function showResult(data, total) {
  const t = data.timing;
  const network = Math.max(0, total - data.server_s);       // phone total minus server time
  const other = Math.max(0, data.server_s - data.waited_s - t.reading_s - t.writing_s);
  const parts = [
    ["Upload + reply (network)", network,       "var(--muted)"],
    ["Waiting for a GPU",        data.waited_s, "var(--err)"],
    ["Reading frames (GPU)",     t.reading_s,   "var(--accent)"],
    ["Writing answer (GPU)",     t.writing_s,   "var(--ok)"],
    ["Other",                    other,         "var(--line)"],
  ];
  $("answer").textContent = data.answer;                      // textContent: show as plain text
  $("result-info").textContent =
    `GPU ${data.gpu} · ${t.input_tokens} input tokens · ${t.answer_tokens} answer tokens · total ${total.toFixed(2)} s`;
  $("bar").innerHTML = parts.map(([, s, color]) =>
    `<span style="width:${(100 * s / total).toFixed(1)}%;background:${color}"></span>`).join("");
  $("parts").innerHTML = parts.map(([name, s, color]) =>
    `<tr><td><span class="swatch" style="background:${color}"></span>${name}</td><td>${s.toFixed(2)} s</td></tr>`).join("");
}

$("send").addEventListener("click", onSend);
