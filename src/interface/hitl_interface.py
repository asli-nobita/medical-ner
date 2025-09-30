"""Web-based Human-in-the-Loop interface (Flask)"""

import os
import json
import uuid
import threading
import webbrowser
import socket
from datetime import datetime
from pathlib import Path

from flask import Flask, send_from_directory, jsonify, request, make_response


def _ensure_dirs():
    os.makedirs("outputs/hitl/session_data", exist_ok=True)
    os.makedirs("outputs/hitl/corrections", exist_ok=True)


def _save_session_files(session_dir, extraction_results, ocr_results):
    extraction_path = os.path.join(session_dir, "extraction_results.json")
    ocr_path = os.path.join(session_dir, "ocr_results.json")
    with open(extraction_path, "w", encoding="utf-8") as f:
        json.dump(extraction_results, f, indent=2, ensure_ascii=False)
    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, indent=2, ensure_ascii=False)
    return extraction_path, ocr_path


def _make_app(static_folder):
    app = Flask(__name__, static_folder=static_folder, static_url_path='')

    @app.after_request
    def _add_cors_headers(resp):
        # Allow simple CORS for local usage
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp

    @app.route("/")
    def index():
        return send_from_directory(static_folder, "index.html")

    @app.route("/<path:filename>")
    def static_files(filename):
        # serve static assets from interface folder
        return send_from_directory(static_folder, filename)

    return app


def _find_free_port(start_port, max_tries=50):
    port = start_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    return None


def start_working_hitl_review(extraction_results, ocr_results, host="127.0.0.1", port=8050, open_browser=True):
    """
    Start a Flask web server that serves the HITL UI and endpoints.
    Writes session files to outputs/hitl/session_data/<session_id>/ and serves index.html.

    Returns:
        dict: {'url': url, 'session_id': session_id, 'session_dir': session_dir, 'thread': thread}
    """
    if not extraction_results:
        print("❌ No extraction results provided.")
        return None

    _ensure_dirs()

    session_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join("outputs", "hitl", "session_data", f"session_{session_id}_{timestamp}")
    Path(session_dir).mkdir(parents=True, exist_ok=True)

    extraction_path, ocr_path = _save_session_files(session_dir, extraction_results, ocr_results)

    interface_dir = os.path.dirname(__file__)  # expects index.html in same folder
    app = _make_app(interface_dir)

    # API endpoints for frontend
    @app.route("/api/session_info", methods=["GET"])
    def session_info():
        return jsonify({
            "session_id": session_id,
            "extraction_file": os.path.relpath(extraction_path),
            "ocr_file": os.path.relpath(ocr_path),
            "session_dir": os.path.relpath(session_dir)
        })

    @app.route("/api/extraction_results", methods=["GET"])
    def get_extraction_results():
        try:
            with open(extraction_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return make_response(jsonify({"error": str(e)}), 500)

    @app.route("/api/ocr_results", methods=["GET"])
    def get_ocr_results():
        try:
            with open(ocr_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return make_response(jsonify({"error": str(e)}), 500)

    @app.route("/api/save_corrections", methods=["POST", "OPTIONS"])
    def save_corrections():
        if request.method == "OPTIONS":
            return make_response(('', 204))
        try:
            payload = request.get_json(force=True)
        except Exception as e:
            return make_response(jsonify({"error": "Invalid JSON payload", "detail": str(e)}), 400)
        if not payload:
            return make_response(jsonify({"error": "No JSON payload provided"}), 400)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"correction_{session_id}_{ts}.json"
        out_path = os.path.join("outputs", "hitl", "corrections", filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return jsonify({"status": "saved", "path": out_path})

    free_port = _find_free_port(port, max_tries=50)
    if free_port is None:
        print(f"❌ No free port found starting at {port}. Cannot start HITL server.")
        return None

    def run_server(host_arg, port_arg):
        try:
            app.run(host=host_arg, port=port_arg, debug=False, use_reloader=False, threaded=True)
        except Exception as e:
            print(f"❌ HITL server failed to start on {host_arg}:{port_arg}: {e}")

    # start server in non-daemon thread so it persists while main thread waits
    thread = threading.Thread(target=run_server, args=(host, free_port), daemon=False)
    thread.start()

    url = f"http://{host}:{free_port}/"
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    print(f"🚀 HITL web UI started at {url}")
    print(f"📁 Session files written to: {session_dir}")
    # return thread so caller can join() or keep reference
    return {"url": url, "session_id": session_id, "session_dir": session_dir, "thread": thread}