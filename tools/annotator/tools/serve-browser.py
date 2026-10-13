#!/usr/bin/env python3
"""Loopback-only CPU annotation bridge. Explicit file allowlists; no uploads.

The native server is the sole source of model/tracking capabilities. The bridge
never fabricates a prediction or silently substitutes another model.
"""
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1] / 'src'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX_BODY = 256 * 1024 * 1024
MAX_CACHE_MIB = 1024 * 1024
ASSETS = {'/': 'index.html', **{'/' + f: f for f in ('index.html', 'style.css', 'app.js', 'bridge.js', 'rle.js', 'coco.js', 'project.js', 'tracking.js')}}


def checked_file(path, allowlist):
    if not isinstance(path, str) or path not in allowlist:
        raise ValueError('file is not in the explicit allowlist')
    target = Path(path)
    if '\n' in path or '\r' in path or str(target.resolve(strict=True)) != path or not target.is_file():
        raise ValueError('allowlisted path changed or is not a file')
    return target


def prompts(args):
    pos, neg, box = args.get('pos', []), args.get('neg', []), args.get('box', args.get('bbox'))
    if not isinstance(pos, list) or not isinstance(neg, list) or len(pos) + len(neg) > 4096:
        raise ValueError('invalid point list')
    def point(value, size):
        if not isinstance(value, list) or len(value) != size or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1e6 for x in value):
            raise ValueError('invalid prompt coordinates')
        return ' '.join(str(x) for x in value)
    commands = ['point ' + point(p, 2) + ' 1' for p in pos]
    commands += ['point ' + point(p, 2) + ' 0' for p in neg]
    if box is not None:
        value = point(box, 4)
        if box[0] >= box[2] or box[1] >= box[3]: raise ValueError('empty/inverted box')
        commands.append('box ' + value)
    if not pos and box is None: raise ValueError('positive point or box is required')
    return commands


def cache_options(enabled=False, directory=None, budget_mib=None):
    if type(enabled) is not bool: raise ValueError('prefix cache flag must be boolean')
    if directory is None:
        if budget_mib is not None: raise ValueError('cache budget requires a persistent cache directory')
        return enabled, None, None
    if type(budget_mib) is not int or not 1 <= budget_mib <= MAX_CACHE_MIB:
        raise ValueError('persistent cache budget must be 1..1048576 MiB')
    path = Path(directory).absolute()
    if any(c in str(path) for c in ('\n', '\r')) or path.resolve(strict=True) != path or not path.is_dir():
        raise ValueError('cache directory must exist without symlinks or traversal')
    # Some managed environments mount empty read-only .git placeholders at
    # workspace roots. A real worktree has a gitfile or .git/HEAD.
    in_worktree = any((parent / '.git').is_file() or (parent / '.git' / 'HEAD').is_file() for parent in (path, *path.parents))
    if '.git' in path.parts or path.is_relative_to(ROOT.parents[2]) or in_worktree:
        raise ValueError('cache directory must be outside Git worktrees')
    return True, path, budget_mib


class Native:
    def __init__(self, executable, models, images, timeout=180, annotation_root=None, experimental_video=False,
                 prefix_cache=False, prefix_cache_dir=None, prefix_cache_budget_mib=None, experimental_q8=False):
        self.executable = str(executable)
        self.experimental_video = bool(experimental_video)
        if type(experimental_q8) is not bool: raise ValueError('experimental Q8 flag must be boolean')
        self.experimental_q8 = experimental_q8
        self.prefix_cache, self.prefix_cache_dir, self.prefix_cache_budget_mib = cache_options(prefix_cache, prefix_cache_dir, prefix_cache_budget_mib)
        self.models, self.images = set(models), set(images)
        self.timeout, self.annotation_root = timeout, annotation_root
        self.process = None
        self.lock = threading.Lock()
        self.process_lock = threading.Lock()
        self.capabilities = {}
        self.generation = 0
        self.reply_buffer = b''
        self.reading_process = None

    def stop(self):
        # Cancellation must not wait for the inference lock.
        with self.process_lock:
            self.generation += 1
            process, self.process = self.process, None
            self.capabilities = {}
        if process:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            if process.stdin: process.stdin.close()
            if process.stdout and self.reading_process is not process: process.stdout.close()
        return None

    def reply(self, process):
        self.reading_process = process
        try: return self._read_reply(process)
        finally:
            self.reading_process = None
            if process and process is not self.process:
                if process.stdin: process.stdin.close()
                if process.stdout: process.stdout.close()

    def _read_reply(self, process):
        if not process: raise ValueError('model is not loaded yet')
        deadline = time.monotonic() + self.timeout
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while b'\n' not in self.reply_buffer:
                if process is not self.process: raise ValueError('native inference was cancelled')
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    self.stop(); raise ValueError('native inference timed out; model stopped')
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk: raise ValueError('sam3_serve exited unexpectedly or was cancelled')
                self.reply_buffer += chunk
                if len(self.reply_buffer) > 128 * 1024 * 1024:
                    self.stop(); raise ValueError('native reply exceeds 128 MiB; model stopped')
        line, self.reply_buffer = self.reply_buffer.split(b'\n', 1)
        value = json.loads(line)
        if not isinstance(value, dict) or value.get('ok') is not True:
            raise ValueError(value.get('error', 'native inference failed') if isinstance(value, dict) else 'invalid native reply')
        if process is not self.process: raise ValueError('native inference was cancelled')
        return value

    def command(self, text):
        process = self.process
        if not process: raise ValueError('model is not loaded yet')
        try:
            process.stdin.write((text + '\n').encode()); process.stdin.flush()
            return self.reply(process)
        except (OSError, json.JSONDecodeError):
            self.stop(); raise ValueError('native inference pipe failed; model stopped')

    def invoke(self, name, args):
        if name == 'sam_stop': return self.stop()
        if name == 'sam_running': return bool(self.process and self.process.poll() is None)
        if name == 'read_image':
            image = checked_file(args.get('path'), self.images)
            if image.stat().st_size > 32 * 1024 * 1024: raise ValueError('image exceeds 32 MiB')
            with image.open('rb') as handle: content = handle.read(32 * 1024 * 1024 + 1)
            if len(content) > 32 * 1024 * 1024: raise ValueError('image exceeds 32 MiB')
            mime = mimetypes.guess_type(str(image))[0] or 'application/octet-stream'
            return {'data_url': 'data:' + mime + ';base64,' + base64.b64encode(content).decode()}
        if name == 'load_saved_project':
            if not self.annotation_root: raise ValueError('no private annotation output root configured')
            root = self.annotation_root
            if root.resolve(strict=True) != root: raise ValueError('annotation root changed')
            path = root / 'annotation-project.json'
            checked_file(str(path), {str(path)})
            if path.stat().st_size > MAX_BODY: raise ValueError('saved project exceeds size limit')
            with path.open('r', encoding='utf-8') as handle: return json.load(handle)
        if name == 'save_project':
            if not self.annotation_root: raise ValueError('no private annotation output root configured')
            project = args.get('project')
            if not isinstance(project, dict) or project.get('schema') != 'sam3-annotation-project' or project.get('version') != 1:
                raise ValueError('invalid project schema')
            root = self.annotation_root
            if root.resolve(strict=True) != root or not root.is_dir(): raise ValueError('annotation root changed')
            # Fixed filename under the CLI-selected private directory; never a
            # client-selected output path. Atomic replace cannot follow symlinks.
            data = json.dumps(project, ensure_ascii=False, allow_nan=False).encode()
            fd, tmp = tempfile.mkstemp(prefix='.annotation-', dir=root)
            try:
                with os.fdopen(fd, 'wb') as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
                os.replace(tmp, root / 'annotation-project.json')
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
            return {'saved': True}
        with self.lock:
            if name == 'sam_start':
                if set(args) - {'modelPath', 'cpu', 'nThreads'}:
                    raise ValueError('runtime/cache configuration is CLI-only')
                model = str(checked_file(args.get('modelPath'), self.models))
                checked_file(self.executable, {self.executable})
                if args.get('cpu') is not True: raise ValueError('this private browser bridge is CPU-only')
                threads = args.get('nThreads', 4)
                if type(threads) is not int or not 1 <= threads <= 32: raise ValueError('threads must be 1..32')
                cache_options(self.prefix_cache, self.prefix_cache_dir, self.prefix_cache_budget_mib)
                self.stop()
                with self.process_lock: generation = self.generation
                self.reply_buffer = b''
                argv = [self.executable, '--model', model, '--cpu', '--n-threads', str(threads)]
                if self.experimental_video: argv.append('--experimental-efficient-video')
                if self.experimental_q8: argv.append('--experimental-efficient-q8-0')
                if self.prefix_cache: argv.append('--efficient-prefix-cache')
                if self.prefix_cache_dir:
                    argv += ['--prefix-cache-dir', str(self.prefix_cache_dir), '--prefix-cache-budget-mib', str(self.prefix_cache_budget_mib)]
                process = subprocess.Popen(argv,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, bufsize=0)
                with self.process_lock:
                    if generation != self.generation:
                        process.terminate(); process.wait(timeout=5)
                        raise ValueError('native model startup was cancelled')
                    self.process = process
                try:
                    result = self.reply(process)
                    def digest(path):
                        with open(path, 'rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()
                    result.update(model_name=Path(model).name, model_sha256=digest(model), native_executable_sha256=digest(self.executable))
                    result['cache_config'] = {'requested': self.prefix_cache, 'persistent': self.prefix_cache_dir is not None, 'budget_mib': self.prefix_cache_budget_mib}
                    with self.process_lock:
                        if process is not self.process: raise ValueError('native model startup was cancelled')
                        self.capabilities = result
                    return result
                except Exception: self.stop(); raise
            if name in ('sam_load_image', 'sam_track_frame'):
                path = str(checked_file(args.get('path'), self.images))
                if name == 'sam_track_frame' and not self.capabilities.get('video_tracking'):
                    raise ValueError('native model has not enabled validated video tracking')
                return self.command(('load ' if name == 'sam_load_image' else 'track_frame ') + path)
            if name == 'sam_track_reset': return self.command('track_reset')
            if name == 'sam_cache_stats':
                if args: raise ValueError('cache statistics accepts no arguments')
                result = self.command('cache_stats')
                fields = ('ram_hits', 'disk_hits', 'misses', 'invalid_entries', 'writes', 'evictions', 'resident_bytes', 'disk_bytes')
                if result.get('event') != 'cache_stats' or type(result.get('enabled')) is not bool or any(type(result.get(key)) is not int or not 0 <= result[key] <= 9007199254740991 for key in fields):
                    raise ValueError('invalid native cache statistics')
                return result
            if name == 'sam_track_config':
                length, start = args.get('sequenceLength'), args.get('sourceStartFrame')
                if type(length) is not int or not 0 <= length <= 2147483647 or type(start) is not int or not 0 <= start <= 2147483647 or (length > 0 and start >= length):
                    raise ValueError('tracking sequence bounds require full length and zero-based seed position')
                if not self.capabilities.get('video_tracking'): raise ValueError('native model has not enabled validated video tracking')
                result = self.command(f'track_config {length} {start}')
                if result.get('event') != 'track_configured' or result.get('sequence_length') != length or result.get('source_start_frame') != start:
                    raise ValueError('native tracking configuration was not acknowledged')
                return result
            if name in ('sam_segment', 'sam_track_add'):
                commands = prompts(args)
                if name == 'sam_track_add':
                    if not self.capabilities.get('video_tracking'): raise ValueError('native model has not enabled validated video tracking')
                    oid = args.get('objectId')
                    if type(oid) is not int or not 1 <= oid <= 2147483647: raise ValueError('object ID must be a positive 32-bit integer')
                self.command('reset')
                for command in commands: self.command(command)
                if name == 'sam_track_add': return self.command('track_add ' + str(oid))
                self.command('multimask ' + ('1' if args.get('multimask') else '0'))
                return self.command('segment')
            raise ValueError('unsupported native command')


class Handler(BaseHTTPRequestHandler):
    def allowed_host(self): return self.headers.get('Host') in self.server.allowed_hosts
    def send_body(self, status, body, mime='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', mime); self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store'); self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        try: self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError): pass
    def do_GET(self):
        if not self.allowed_host(): return self.send_body(403, b'Forbidden')
        name = urlparse(self.path).path
        if name not in ASSETS: return self.send_body(404, b'Not found')
        file = ROOT / ASSETS[name]; body = file.read_bytes()
        if file.suffix == '.html':
            config = json.dumps(self.server.config, ensure_ascii=True).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
            body = body.replace(b'<script src="bridge.js">', ('<script>window.__SAM3_LOOPBACK__=' + config + ';</script>\n<script src="bridge.js">').encode())
        self.send_body(200, body, mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
    def do_POST(self):
        if not self.allowed_host() or self.headers.get('Origin') not in self.server.origins:
            return self.send_body(403, b'{"error":"same-origin request required"}')
        if self.path != '/api/invoke' or self.headers.get('Content-Type','').split(';')[0] != 'application/json':
            return self.send_body(415, b'{"error":"JSON API required"}')
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY: raise ValueError('invalid body size')
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict) or not isinstance(request.get('args', {}), dict): raise ValueError('invalid request')
            value = self.server.native.invoke(request.get('command'), request.get('args', {}))
            self.send_body(200, json.dumps({'result': value}, allow_nan=False).encode())
        except (ValueError, OSError, KeyError, TypeError) as error:
            self.send_body(400, json.dumps({'error': str(error)}).encode())
    def log_message(self, format, *args): pass


def catalog(roots):
    groups = {}
    for value in roots:
        root = value.resolve(strict=True)
        if not root.is_dir(): raise ValueError('image root must be a directory')
        for image in sorted(root.rglob('*')):
            if image.is_file() and image.suffix.lower() in EXTENSIONS:
                resolved = image.resolve()
                # Symlinks are excluded, even when they point inside the root.
                if resolved == image and resolved.is_relative_to(root):
                    groups.setdefault(image.parent, {})[str(image)] = {'path': str(image), 'key': image.name, 'fileName': image.name}
    sequences = [{'id': f'{index + 1}:{parent.name}', 'frames': list(images.values())} for index, (parent, images) in enumerate(sorted(groups.items()))]
    images = sorted({f['path'] for s in sequences for f in s['frames']})
    if not images or len(images) > 20000: raise ValueError('image allowlist must contain 1..20000 files')
    return images, sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve', type=Path, required=True)
    parser.add_argument('--model', type=Path, action='append', required=True)
    parser.add_argument('--image-root', type=Path, action='append', required=True)
    parser.add_argument('--annotation-root', type=Path, help='Existing private directory outside the repository for atomic project saves')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--experimental-efficient-video', action='store_true', help='Compatibility flag for older native servers; qualified base-plus video no longer requires it')
    parser.add_argument('--experimental-efficient-q8-0', action='store_true', help='Explicit unqualified Q8 compute opt-in; never enabled by the browser')
    parser.add_argument('--efficient-prefix-cache', action='store_true', help='Enable bounded one-entry RAM prefix cache')
    parser.add_argument('--prefix-cache-dir', type=Path, help='Existing private persistent cache directory outside Git, without symlinks')
    parser.add_argument('--prefix-cache-budget-mib', type=int, help='Explicit persistent cache budget, 1..1048576 MiB; requires --prefix-cache-dir')
    args = parser.parse_args()
    try: prefix_cache, cache_dir, cache_budget = cache_options(args.efficient_prefix_cache, args.prefix_cache_dir, args.prefix_cache_budget_mib)
    except (ValueError, OSError) as error: parser.error(str(error))
    executable = args.serve.resolve(strict=True)
    models = [str(path.resolve(strict=True)) for path in args.model]
    images, sequences = catalog(args.image_root)
    for path in models + images:
        if '\n' in path or '\r' in path: parser.error('newline in a protocol path')
    output = args.annotation_root.resolve(strict=True) if args.annotation_root else None
    if output and (not output.is_dir() or output.is_relative_to(ROOT.parents[2])):
        parser.error('--annotation-root must be an existing directory outside this repository')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    port = server.server_address[1]
    server.allowed_hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
    server.origins = {'http://' + host for host in server.allowed_hosts}
    server.native = Native(executable, models, images, annotation_root=output, experimental_video=args.experimental_efficient_video,
        experimental_q8=args.experimental_efficient_q8_0, prefix_cache=prefix_cache, prefix_cache_dir=cache_dir, prefix_cache_budget_mib=cache_budget)
    server.config = {'models': models, 'images': images, 'sequences': sequences, 'cpuOnly': True, 'canSave': bool(output)}
    print(f'Private CPU annotator: http://127.0.0.1:{port} ({len(images)} allowlisted images)', flush=True)
    try: server.serve_forever()
    finally: server.native.stop(); server.server_close()

if __name__ == '__main__': main()
