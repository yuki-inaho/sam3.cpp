// SAM-assisted annotation shell.
//
// The window is plain HTML/JS; everything that needs the filesystem or the
// model lives here.  Segmentation runs in a long-lived `sam3_serve` child
// process: it encodes the image once (seconds) and then answers prompts in
// roughly half a second, which is what makes click-to-segment usable.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use base64::Engine;
use serde::Serialize;
use serde_json::Value;
use tauri::Manager;

struct Server {
    child:  Child,
    stdin:  ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Server {
    // Every command answers with exactly one JSON line, so request and reply
    // stay in lockstep and no framing is needed.
    fn request(&mut self, line: &str) -> Result<Value, String> {
        writeln!(self.stdin, "{}", line).map_err(|e| format!("write to sam3_serve: {e}"))?;
        self.stdin.flush().map_err(|e| format!("flush sam3_serve: {e}"))?;
        self.read_reply()
    }

    fn read_reply(&mut self) -> Result<Value, String> {
        let mut buf = String::new();
        let n = self
            .stdout
            .read_line(&mut buf)
            .map_err(|e| format!("read from sam3_serve: {e}"))?;
        if n == 0 {
            return Err("sam3_serve exited unexpectedly".into());
        }
        let v: Value = serde_json::from_str(buf.trim())
            .map_err(|e| format!("bad reply from sam3_serve: {e} (line: {buf})"))?;
        if v.get("ok").and_then(Value::as_bool) != Some(true) {
            let msg = v
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            return Err(msg.to_string());
        }
        Ok(v)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

#[derive(Default)]
struct AppState {
    server: Mutex<Option<Server>>,
}

#[derive(Serialize)]
struct LoadedImage {
    data_url: String,
    width:    u32,
    height:   u32,
}

// Look for sam3_serve next to the app first, then in the usual build trees, so
// `cargo run` from a checkout works without configuration.
// Which executable to run is deliberately NOT a parameter of the command: the
// webview must not be able to name an arbitrary binary for the app to spawn.
// The search is the app's own directory, then the build tree relative to that
// directory (not to the process's working directory, which depends on where
// the app happened to be launched from), with an explicit environment override
// for developers.
fn find_serve_binary() -> Result<PathBuf, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("sam3_serve"));
            // target/{debug,release} -> repo root -> build/examples
            for up in ["../../../..", "../../../../.."] {
                candidates.push(dir.join(up).join("build/examples/sam3_serve"));
            }
        }
    }
    if let Ok(p) = std::env::var("SAM3_SERVE") {
        candidates.insert(0, PathBuf::from(p));
    }

    for c in &candidates {
        if c.is_file() {
            return c.canonicalize().map_err(|e| e.to_string());
        }
    }
    Err(format!(
        "sam3_serve not found. Build it with `cmake --build build --target sam3_serve`, \
         or set SAM3_SERVE. Looked in: {}",
        candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    ))
}

#[tauri::command]
fn sam_start(
    state: tauri::State<'_, AppState>,
    model_path: String,
    cpu: bool,
    n_threads: Option<u32>,
) -> Result<Value, String> {
    check_path(&model_path)?;
    if !Path::new(&model_path).is_file() {
        return Err(format!("model not found: {model_path}"));
    }
    let exe = find_serve_binary()?;

    let mut cmd = Command::new(&exe);
    cmd.arg("--model").arg(&model_path);
    if cpu {
        cmd.arg("--cpu");
    }
    if let Some(n) = n_threads {
        cmd.arg("--n-threads").arg(n.to_string());
    }
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        // sam3_serve is chatty on stderr; if nothing drains it the pipe fills
        // and the child blocks mid-inference, so send it to the console.
        .stderr(Stdio::inherit());

    let mut child = cmd.spawn().map_err(|e| format!("spawn {}: {e}", exe.display()))?;
    let stdin = child.stdin.take().ok_or("no stdin on sam3_serve")?;
    let stdout = BufReader::new(child.stdout.take().ok_or("no stdout on sam3_serve")?);

    let mut server = Server { child, stdin, stdout };
    // Model loading happens before the banner, so this read is the handshake.
    // On failure `server` drops here, reaping the child we just spawned.
    let ready = server.read_reply()?;

    // Install only after the handshake succeeded, and drop whatever was there
    // first so two model processes never hold weights at the same time.
    let mut guard = state.server.lock().unwrap();
    *guard = None;
    *guard = Some(server);
    Ok(ready)
}

#[tauri::command]
fn sam_stop(state: tauri::State<'_, AppState>) -> Result<(), String> {
    *state.server.lock().unwrap() = None; // Drop sends `quit` and reaps
    Ok(())
}

#[tauri::command]
fn sam_running(state: tauri::State<'_, AppState>) -> bool {
    state.server.lock().unwrap().is_some()
}

// A path is spliced into a newline-framed protocol line, so a newline in it
// would inject a second command and desynchronize every later request/reply
// pair. Reject it here rather than corrupting the stream.
fn check_path(path: &str) -> Result<(), String> {
    if path.contains('\n') || path.contains('\r') {
        return Err("path contains a newline, which the sam3_serve protocol cannot carry".into());
    }
    Ok(())
}

// Once a request fails the child may be gone or the stream out of step; either
// way the Server is not usable again, so drop it instead of leaving a dead one
// installed that sam_running would keep reporting as alive.
fn with_server<T>(
    state: &tauri::State<'_, AppState>,
    f: impl FnOnce(&mut Server) -> Result<T, String>,
) -> Result<T, String> {
    let mut guard = state.server.lock().unwrap();
    let server = guard.as_mut().ok_or("model is not loaded yet")?;
    match f(server) {
        Ok(v) => Ok(v),
        Err(e) => {
            *guard = None;
            Err(e)
        }
    }
}

#[tauri::command]
fn sam_load_image(state: tauri::State<'_, AppState>, path: String) -> Result<Value, String> {
    check_path(&path)?;
    with_server(&state, |server| server.request(&format!("load {path}")))
}

#[tauri::command]
fn sam_segment(
    state: tauri::State<'_, AppState>,
    pos: Vec<[f32; 2]>,
    neg: Vec<[f32; 2]>,
    bbox: Option<[f32; 4]>,
    multimask: bool,
) -> Result<Value, String> {
    // Coordinates are finite floats or the formatted line is not parseable by
    // the server's istringstream, which would leave the stream mid-command.
    let finite = pos.iter().chain(neg.iter()).all(|p| p.iter().all(|v| v.is_finite()))
        && bbox.map_or(true, |b| b.iter().all(|v| v.is_finite()));
    if !finite {
        return Err("prompt contains a non-finite coordinate".into());
    }

    with_server(&state, |server| {
        server.request("reset")?;
        for p in &pos {
            server.request(&format!("point {} {} 1", p[0], p[1]))?;
        }
        for p in &neg {
            server.request(&format!("point {} {} 0", p[0], p[1]))?;
        }
        if let Some(b) = bbox {
            server.request(&format!("box {} {} {} {}", b[0], b[1], b[2], b[3]))?;
        }
        if multimask {
            server.request("multimask 1")?;
        }
        server.request("segment")
    })
}

// The webview cannot read arbitrary local files, so hand it a data URL.
#[tauri::command]
fn read_image(path: String) -> Result<LoadedImage, String> {
    check_path(&path)?;
    let bytes = std::fs::read(&path).map_err(|e| format!("read {path}: {e}"))?;
    let mime = match Path::new(&path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => "image/png",
        Some("bmp") => "image/bmp",
        Some("webp") => "image/webp",
        _ => "image/jpeg",
    };
    let (width, height) = image_size(&bytes).unwrap_or((0, 0));
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(LoadedImage { data_url: format!("data:{mime};base64,{b64}"), width, height })
}

// Enough header parsing to report dimensions without pulling in an image crate.
fn image_size(b: &[u8]) -> Option<(u32, u32)> {
    if b.len() > 24 && b.starts_with(&[0x89, b'P', b'N', b'G']) {
        let w = u32::from_be_bytes([b[16], b[17], b[18], b[19]]);
        let h = u32::from_be_bytes([b[20], b[21], b[22], b[23]]);
        return Some((w, h));
    }
    if b.len() > 4 && b[0] == 0xFF && b[1] == 0xD8 {
        let mut i = 2usize;
        while i + 1 < b.len() {
            if b[i] != 0xFF {
                i += 1;
                continue;
            }
            let marker = b[i + 1];
            // 0xFF is legal padding before a marker; RSTn/SOI/EOI/TEM carry no
            // length field, so reading two bytes there would be garbage and
            // i += 2 + len would jump somewhere arbitrary.
            if marker == 0xFF {
                i += 1;
                continue;
            }
            if marker == 0x01 || (0xD0..=0xD9).contains(&marker) {
                i += 2;
                continue;
            }
            if i + 3 >= b.len() {
                break;
            }
            let len = u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize;
            if len < 2 {
                break; // malformed: would not advance
            }
            // SOF0..SOF15, excluding the non-frame DHT/JPG/DAC markers
            if (0xC0..=0xCF).contains(&marker) && marker != 0xC4 && marker != 0xC8 && marker != 0xCC {
                if i + 8 >= b.len() {
                    break;
                }
                let h = u16::from_be_bytes([b[i + 5], b[i + 6]]) as u32;
                let w = u16::from_be_bytes([b[i + 7], b[i + 8]]) as u32;
                return Some((w, h));
            }
            i += 2 + len;
        }
    }
    None
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            sam_start,
            sam_stop,
            sam_running,
            sam_load_image,
            sam_segment,
            read_image
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                // Drop the child rather than leaving an orphan holding the model
                if let Some(state) = window.try_state::<AppState>() {
                    *state.server.lock().unwrap() = None;
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("failed to start the annotator");
}
