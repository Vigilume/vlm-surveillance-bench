const $ = (id) => document.getElementById(id);

const N_FRAMES = 8;                       // frames per segment
const WINDOW_S = 6;                       // seconds per segment
const GAP_S = WINDOW_S / N_FRAMES;        // 0.75 s between frames
const MAX_SIDE = 512;                     // longest side of each frame
const MAX_ON_THE_WAY = 2;                 // at most 2 segments waiting for an answer
const MAX_CARDS = 20;                     // keep only the newest 20 cards

let stream = null, timer = null, current = [], onTheWay = 0, segmentNo = 0;
let counts = { sent: 0, answered: 0, skipped: 0, dropped: 0 };
const canvas = document.createElement("canvas");

// --- Status pill ---
async function checkServer() {
  try {
    const data = await (await fetch("hello?name=live")).json();
    $("status").className = "pill ok";
    $("status-text").textContent = `connected · ${data.gpus.length} GPUs`;
  } catch (error) {
    $("status").className = "pill err";
    $("status-text").textContent = "offline";
  }
}

function showCounters() {
  $("counters").textContent = `sent ${counts.sent} · answered ${counts.answered} · ` +
    `skipped on phone ${counts.skipped} · dropped by server ${counts.dropped} · on the way ${onTheWay}`;
}

// --- Small helper: make an element with a class and text ---
function element(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

// --- Put a card at the top of the feed; remove the oldest beyond MAX_CARDS ---
function addToFeed(card) {
  const feed = $("feed");
  feed.prepend(card);
  while (feed.children.length > MAX_CARDS) {
    const oldest = feed.lastElementChild;
    oldest.querySelectorAll("img").forEach((img) => URL.revokeObjectURL(img.src));   // free memory
    oldest.remove();
  }
}

function addNote(no, text) {                 // grey card: skipped, dropped, error
  const card = element("div", "seg note");
  card.append(element("p", "answer", `#${no} · ${text}`));
  addToFeed(card);
}

function addAnswer(no, frames, data, roundTrip, shortest, longest) {
  const card = element("div", "seg");
  card.append(element("p", "answer", data.answer));
  const meta = element("div", "meta");
  meta.append(
    element("span", "tag", `#${no}`),
    element("span", `tag gpu${data.gpu}`, `GPU ${data.gpu}`),
    element("span", "tag", `round trip ${roundTrip.toFixed(1)} s`),
    element("span", "tag", `delay ${shortest.toFixed(1)}–${longest.toFixed(1)} s`),
    element("span", "tag", `${data.timing.answer_tokens} tokens`),
  );
  const strip = element("div", "strip");
  for (const f of frames) {
    const img = element("img");
    img.src = URL.createObjectURL(f.blob);
    strip.append(img);
  }
  card.append(meta, strip);
  addToFeed(card);
}

// --- Copy the current camera picture into a JPEG ---
function grabFrame() {
  const video = $("camera");
  const scale = Math.min(1, MAX_SIDE / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
}

// --- Called every 0.75 s ---
async function tick() {
  const blob = await grabFrame();
  current.push({ blob, at: performance.now() });
  if (current.length === N_FRAMES) {
    const frames = current;
    current = [];
    segmentNo += 1;
    sendSegment(segmentNo, frames);          // no await: keep collecting
  }
}

// --- Send one segment, show its card when the answer arrives ---
async function sendSegment(no, frames) {
  if (onTheWay >= MAX_ON_THE_WAY) {
    counts.skipped += 1;
    addNote(no, "skipped on the phone (2 segments still on their way)");
    showCounters();
    return;
  }
  onTheWay += 1;
  counts.sent += 1;
  showCounters();

  const form = new FormData();
  frames.forEach((f, k) => form.append("frames", f.blob, `frame${k}.jpg`));
  form.append("gap", GAP_S.toString());
  form.append("short", $("short").checked ? "true" : "false");
  const sentAt = performance.now();

  try {
    const reply = await fetch("analyze_frames", { method: "POST", body: form });
    const data = await reply.json();
    const now = performance.now();
    if (!reply.ok) throw new Error(data.error || `status ${reply.status}`);
    if (data.dropped) {
      counts.dropped += 1;
      addNote(no, "dropped by the server (a newer segment replaced it)");
    } else {
      counts.answered += 1;
      addAnswer(no, frames, data,
                (now - sentAt) / 1000,                              // round trip
                (now - frames[frames.length - 1].at) / 1000,        // delay since the last frame
                (now - frames[0].at) / 1000);                       // delay since the first frame
    }
  } catch (error) {
    addNote(no, "error: " + error.message);
  }
  onTheWay -= 1;
  showCounters();
}

// --- Start / stop ---
async function startLive() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    const video = $("camera");
    video.srcObject = stream;
    await video.play();
    $("cam-info").textContent = `Camera on · ${video.videoWidth}×${video.videoHeight} px · ` +
      `1 frame every ${GAP_S} s · ${N_FRAMES} frames per segment`;
    current = [];
    timer = setInterval(tick, GAP_S * 1000);
    $("start").disabled = true;
    $("stop").disabled = false;
    showCounters();
  } catch (error) {
    $("cam-info").textContent = "Camera error: " + error.message;
  }
}

function stopLive() {
  clearInterval(timer);
  timer = null;
  current = [];
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  $("camera").srcObject = null;
  $("cam-info").textContent = "Camera off.";
  $("start").disabled = false;
  $("stop").disabled = true;
}

$("start").addEventListener("click", startLive);
$("stop").addEventListener("click", stopLive);
checkServer();
