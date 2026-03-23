import base64
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - import guard for local launch
    raise SystemExit(
        "Gradio is not installed. Install dependencies from requirements.txt, then run this file again."
    ) from exc

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
MAX_LOG_LINES = 600
PREVIEW_MAX_FRAMES = 80
PREVIEW_FPS = 8
OPENAI_API_URL = "https://api.openai.com/v1/responses"
RENDER_FRAME_PATTERN = re.compile(
    r"^(?P<prefix>.+)_step(?P<step>\d+)(?P<tail>(?:_.+)?)\.(png|jpg|jpeg)$",
    re.IGNORECASE,
)
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def trim_logs(log_lines):
    if len(log_lines) > MAX_LOG_LINES:
        del log_lines[: len(log_lines) - MAX_LOG_LINES]


def logs_text(log_lines):
    return "\n".join(log_lines) if log_lines else "No logs yet."


def parse_camera_list(raw_value, key_name):
    parts = [part for part in str(raw_value).replace(",", " ").split() if part]
    if len(parts) < 2:
        raise ValueError(f"{key_name} must contain at least two camera indices.")
    return parts


def derive_model_path(dataset_dir, model_dir):
    explicit = str(model_dir or "").strip()
    if explicit:
        return explicit
    dataset_name = Path(str(dataset_dir).rstrip("\\/")).name.strip()
    if not dataset_name:
        raise ValueError("Unable to infer model directory from dataset_dir.")
    return str(Path("output") / dataset_name)


def open_directory_dialog(initial_dir):
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("Tkinter is not available for native folder selection.") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(initialdir=initial_dir or str(ROOT_DIR))
    finally:
        root.destroy()
    return selected or None


def open_file_dialog(initial_dir, file_types):
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("Tkinter is not available for native file selection.") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askopenfilename(
            initialdir=initial_dir or str(ROOT_DIR),
            filetypes=file_types,
        )
    finally:
        root.destroy()
    return selected or None


def browse_dataset_directory(current_dataset_dir):
    current = str(current_dataset_dir or "").strip()
    initial_dir = current if current and Path(current).exists() else str(ROOT_DIR)
    selected = open_directory_dialog(initial_dir)
    return selected or current


def browse_reference_image(dataset_dir, current_reference_image):
    current_image = str(current_reference_image or "").strip()
    initial_dir = None

    if current_image:
        current_path = Path(current_image)
        if current_path.exists():
            initial_dir = str(current_path.parent)

    if initial_dir is None:
        try:
            initial_dir = str(find_dataset_image_dir(dataset_dir))
        except Exception:
            dataset_path = str(dataset_dir or "").strip()
            initial_dir = dataset_path if dataset_path and Path(dataset_path).exists() else str(ROOT_DIR)

    selected = open_file_dialog(
        initial_dir,
        [
            ("Image Files", "*.png *.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("JPEG", "*.jpg *.jpeg"),
            ("All Files", "*.*"),
        ],
    )
    chosen = selected or current_image
    preview = preview_selected_reference_image(chosen)
    return chosen, preview


def find_dataset_image_dir(dataset_dir):
    dataset_path = Path(str(dataset_dir).strip())
    if not dataset_path.is_dir():
        raise ValueError(f"Dataset directory not found: {dataset_path}")

    for candidate in ("images", "color", "input"):
        image_dir = dataset_path / candidate
        if image_dir.is_dir():
            return image_dir

    raise ValueError(f"No images/color/input directory found under {dataset_path}")


def parse_json_loose(raw_text):
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Model returned empty output.")

    match = JSON_BLOCK_PATTERN.search(text)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_positions = [idx for idx in (text.find("["), text.find("{")) if idx != -1]
        if not start_positions:
            raise
        start = min(start_positions)
        end = max(text.rfind("]"), text.rfind("}"))
        if end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def resolve_api_key(api_key):
    candidate = str(api_key or "").strip()
    if candidate:
        return candidate
    candidate = os.getenv("OPENAI_API_KEY", "").strip()
    if candidate:
        return candidate
    raise ValueError("OpenAI API key is required. Enter it in the password field or set OPENAI_API_KEY.")


def build_image_data_url(image_path):
    image_bytes = image_path.read_bytes()
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def call_openai_vision(api_key, model_name, prompt_text, image_path):
    payload = {
        "model": model_name,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {
                        "type": "input_image",
                        "image_url": build_image_data_url(image_path),
                    },
                ],
            }
        ],
    }

    request = urllib_request.Request(
        OPENAI_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=90) as response:
            raw = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed ({exc.code}): {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

    response_json = json.loads(raw)
    output_text = str(response_json.get("output_text", "")).strip()
    if output_text:
        return output_text

    output_parts = []
    for item in response_json.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                output_parts.append(content["text"])

    combined = "\n".join(output_parts).strip()
    if not combined:
        raise RuntimeError("OpenAI API response did not include text output.")
    return combined


def make_foreground_prompt():
    return (
        "You are analyzing a single scene image for a 3D semantic editing pipeline. "
        "Return ONLY a valid JSON array of strings. Each string must be one candidate "
        "foreground object group written as a comma-separated list of 1 to 4 short noun phrases. "
        "If the image contains multiple distinct candidate target groups, return multiple strings. "
        "Only group items together when they are adjacent and should be treated as one selectable object set "
        "(for example: 'vase,flowers'). Prefer concrete visible objects. Keep all phrases lowercase. "
        "Do not include explanations, markdown, numbering, or any extra keys."
    )


def make_autofill_prompt(selected_foreground):
    return (
        "You are helping configure a 3D semantic physics editing UI from a single scene image. "
        f"The user selected this foreground object group: '{selected_foreground}'. "
        "Return ONLY a valid JSON object with exactly these string keys: "
        "'text_query', 'foreground_objects', 'background_objects', 'ground_plane', 'rigid_object'. "
        "'text_query' should be the best short phrase for a CLIP heatmap query. "
        "'foreground_objects' should be a normalized comma-separated list matching one editable object set. "
        "'background_objects' should be a comma-separated list of nearby distractors to exclude. "
        "'ground_plane' should be one short surface label such as 'tabletop', 'desk', 'floor', 'shelf', or an empty string if unclear. "
        "'rigid_object' should be one short noun phrase for the rigid subset of the selected foreground, or an empty string if not useful. "
        "Keep all phrases lowercase. Do not include markdown, comments, or extra keys."
    )


def preview_selected_reference_image(reference_image_path):
    image_path = str(reference_image_path or "").strip()
    if not image_path:
        return None
    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"Selected image not found: {path}")
    return str(path)


def suggest_foreground_candidates(api_key, model_name, reference_image_path):
    api_token = resolve_api_key(api_key)
    image_path_str = str(reference_image_path or "").strip()
    if not image_path_str:
        raise ValueError("Select a reference image first.")
    image_path = Path(image_path_str)
    if not image_path.is_file():
        raise ValueError(f"Selected image not found: {image_path}")

    raw_text = call_openai_vision(api_token, str(model_name).strip() or "gpt-5.2", make_foreground_prompt(), image_path)
    parsed = parse_json_loose(raw_text)
    if not isinstance(parsed, list):
        raise ValueError("Foreground suggestion response was not a JSON list.")

    candidates = []
    for item in parsed:
        text = str(item).strip()
        if text:
            candidates.append(text)
    if not candidates:
        raise ValueError("No valid foreground candidates returned by the model.")

    status = (
        f"### Foreground Candidates Ready\n"
        f"- Image: `{image_path.name}`\n"
        f"- Candidates: `{len(candidates)}`"
    )
    return (
        status,
        raw_text,
        gr.update(choices=candidates, value=candidates[0]),
    )


def autofill_query_fields(api_key, model_name, reference_image_path, selected_foreground):
    api_token = resolve_api_key(api_key)
    image_path_str = str(reference_image_path or "").strip()
    foreground_choice = str(selected_foreground or "").strip()
    if not image_path_str:
        raise ValueError("Select a reference image first.")
    if not foreground_choice:
        raise ValueError("Select one foreground candidate first.")

    image_path = Path(image_path_str)
    if not image_path.is_file():
        raise ValueError(f"Selected image not found: {image_path}")

    raw_text = call_openai_vision(
        api_token,
        str(model_name).strip() or "gpt-5.2",
        make_autofill_prompt(foreground_choice),
        image_path,
    )
    parsed = parse_json_loose(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("Autofill response was not a JSON object.")

    text_query = str(parsed.get("text_query", "")).strip()
    foreground_objects = str(parsed.get("foreground_objects", foreground_choice)).strip() or foreground_choice
    background_objects = str(parsed.get("background_objects", "")).strip()
    ground_plane = str(parsed.get("ground_plane", "")).strip()
    rigid_object = str(parsed.get("rigid_object", "")).strip()

    status = (
        f"### Query Fields Autofilled\n"
        f"- Image: `{image_path.name}`\n"
        f"- Foreground: `{foreground_objects}`"
    )
    return (
        status,
        raw_text,
        text_query,
        foreground_objects,
        background_objects,
        ground_plane,
        rigid_object,
    )


def find_latest_iteration(model_path):
    point_cloud_dir = Path(model_path) / "point_cloud"
    if not point_cloud_dir.is_dir():
        raise ValueError(f"point_cloud directory not found under {model_path}")

    latest = None
    for entry in point_cloud_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("iteration_"):
            continue
        try:
            iteration = int(entry.name.split("_", 1)[1])
        except ValueError:
            continue
        if latest is None or iteration > latest:
            latest = iteration

    if latest is None:
        raise ValueError(f"No saved iterations found under {point_cloud_dir}")
    return latest


def render_directory_for_model(model_path):
    iteration = find_latest_iteration(model_path)
    return Path(model_path) / "interpolating_camera" / f"ours_{iteration}" / "renders"


def collect_render_frames(frame_kind, render_dir, started_at):
    if not render_dir.is_dir():
        raise RuntimeError(f"Render directory not found: {render_dir}")

    matched = []
    for path in render_dir.iterdir():
        if not path.is_file():
            continue
        try:
            modified_at = path.stat().st_mtime
        except OSError:
            continue
        if modified_at + 1e-6 < started_at:
            continue

        match = RENDER_FRAME_PATTERN.match(path.name)
        if not match:
            continue

        tail = match.group("tail").lower()
        if frame_kind == "heatmap":
            if tail != "_heatmap":
                continue
        elif frame_kind == "final":
            if tail:
                continue
        else:
            raise ValueError(f"Unknown frame kind: {frame_kind}")

        matched.append(
            (
                match.group("prefix").lower(),
                int(match.group("step")),
                tail,
                path.name,
            )
        )

    matched.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in matched]


def build_preview_gif(render_dir, frame_names, output_name):
    if not frame_names:
        raise ValueError("No frames available for preview GIF.")

    stride = max(1, math.ceil(len(frame_names) / PREVIEW_MAX_FRAMES))
    selected_names = frame_names[::stride]
    if selected_names[-1] != frame_names[-1]:
        selected_names.append(frame_names[-1])

    frames = []
    for file_name in selected_names:
        frame_path = render_dir / file_name
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB"))

    output_path = render_dir / output_name
    frame_duration_ms = max(1, int(1000 / PREVIEW_FPS))
    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    return str(output_path)


def frame_gallery(render_dir, frame_names):
    return [str(render_dir / file_name) for file_name in frame_names]


def run_command_stream(step_name, command, log_lines):
    log_lines.append(f"[{step_name}]")
    log_lines.append(f"$ {' '.join(command)}")
    trim_logs(log_lines)
    yield None

    process = subprocess.Popen(
        command,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        if text:
            log_lines.append(text)
            trim_logs(log_lines)
            yield None

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{step_name} failed with exit code {return_code}")


def make_status(message, model_path, state):
    title = f"### {message}"
    details = [
        f"- Model path: `{model_path}`",
        f"- State: `{state}`",
    ]
    return "\n".join([title, *details])


def basic_updates(status, model_path, log_lines, state):
    return (
        make_status(status, model_path, state),
        model_path,
        logs_text(log_lines),
    )


def query_updates(status, model_path, log_lines, heatmap_gif, heatmap_gallery_paths, final_gif, final_gallery_paths, state):
    return (
        make_status(status, model_path, state),
        model_path,
        logs_text(log_lines),
        heatmap_gif,
        heatmap_gallery_paths,
        final_gif,
        final_gallery_paths,
    )


def collect_heatmap_outputs(model_path, render_started_at, log_lines):
    render_dir = render_directory_for_model(model_path)
    frame_names = collect_render_frames("heatmap", render_dir, render_started_at)
    if not frame_names:
        raise RuntimeError(f"No heatmap frames found in {render_dir}")
    preview_path = build_preview_gif(render_dir, frame_names, "ui_heatmap_preview.gif")
    log_lines.append(f"Collected {len(frame_names)} heatmap frames.")
    trim_logs(log_lines)
    return preview_path, frame_gallery(render_dir, frame_names)


def collect_final_outputs(model_path, render_started_at, log_lines):
    render_dir = render_directory_for_model(model_path)
    frame_names = collect_render_frames("final", render_dir, render_started_at)
    if not frame_names:
        raise RuntimeError(f"No final RGB frames found in {render_dir}")
    preview_path = build_preview_gif(render_dir, frame_names, "ui_final_preview.gif")
    log_lines.append(f"Collected {len(frame_names)} final frames.")
    trim_logs(log_lines)
    return preview_path, frame_gallery(render_dir, frame_names)


def build_convert_command(dataset_dir, convert_skip_matching, convert_resize):
    command = [
        sys.executable,
        "convert.py",
        "-s",
        dataset_dir,
    ]
    if convert_skip_matching:
        command.append("--skip_matching")
    if convert_resize:
        command.append("--resize")
    return command


def build_feature_command(dataset_dir):
    return [
        sys.executable,
        "compute_obj_part_feature.py",
        "-s",
        dataset_dir,
    ]


def build_train_command(dataset_dir, model_path, train_iterations):
    return [
        sys.executable,
        "train.py",
        "-s",
        dataset_dir,
        "-m",
        model_path,
        "--iterations",
        str(train_iterations),
    ]


def build_heatmap_command(dataset_dir, model_path, heatmap_camera_slerp_list, heatmap_step_size, text_query, neg_text_query):
    command = [
        sys.executable,
        "render.py",
        "-m",
        model_path,
        "-s",
        dataset_dir,
        "--camera_slerp_list",
        *parse_camera_list(heatmap_camera_slerp_list, "heatmap_camera_slerp_list"),
        "--with_feat",
        "--clip_feat",
        "--text_query",
        text_query.strip(),
        "--step_size",
        str(int(heatmap_step_size)),
    ]
    if str(neg_text_query).strip():
        command.extend(["--neg_text_query", str(neg_text_query).strip()])
    return command


def build_segment_command(model_path, fg_obj_list, bg_obj_list, ground_plane_name, rigid_object_name,
                          threshold, object_select_eps, inward_bbox_offset, final_noise_filtering):
    command = [
        sys.executable,
        "segment.py",
        "-m",
        model_path,
        "--fg_obj_list",
        fg_obj_list.strip(),
        "--bg_obj_list",
        bg_obj_list.strip(),
        "--ground_plane_name",
        ground_plane_name.strip(),
        "--threshold",
        str(float(threshold)),
        "--object_select_eps",
        str(float(object_select_eps)),
        "--inward_bbox_offset",
        str(float(inward_bbox_offset)),
    ]
    if str(rigid_object_name).strip():
        command.extend(["--rigid_object_name", str(rigid_object_name).strip()])
    if final_noise_filtering:
        command.append("--final_noise_filtering")
    return command


def build_physics_command(model_path, rigid_speed, use_rigidity):
    command = [
        sys.executable,
        "mpm_physics.py",
        "-m",
        model_path,
        "--rigid_speed",
        str(float(rigid_speed)),
        "--headless",
    ]
    if use_rigidity:
        command.append("--use_rigidity")
    return command


def build_final_render_command(dataset_dir, model_path, final_camera_slerp_list, final_step_size):
    return [
        sys.executable,
        "render.py",
        "-m",
        model_path,
        "-s",
        dataset_dir,
        "--camera_slerp_list",
        *parse_camera_list(final_camera_slerp_list, "final_camera_slerp_list"),
        "--step_size",
        str(int(final_step_size)),
        "--with_editing",
    ]


def setup_runner(
    run_convert,
    convert_skip_matching,
    convert_resize,
    run_feature_compute,
    run_training,
    dataset_dir,
    model_dir,
    train_iterations,
):
    log_lines = []

    try:
        dataset_dir = str(dataset_dir).strip()
        if not dataset_dir:
            raise ValueError("dataset_dir is required.")

        model_path = derive_model_path(dataset_dir, model_dir)
        log_lines.append("Setup request accepted.")
        log_lines.append(f"Resolved model path: {model_path}")
        trim_logs(log_lines)
        yield basic_updates("Preparing setup stages", model_path, log_lines, "running")

        if run_convert:
            for _ in run_command_stream(
                "COLMAP convert",
                build_convert_command(dataset_dir, convert_skip_matching, convert_resize),
                log_lines,
            ):
                yield basic_updates("Running COLMAP convert", model_path, log_lines, "running")

        if run_feature_compute:
            for _ in run_command_stream(
                "Compute features",
                build_feature_command(dataset_dir),
                log_lines,
            ):
                yield basic_updates("Computing features", model_path, log_lines, "running")

        if run_training:
            for _ in run_command_stream(
                "Train feature splatting",
                build_train_command(dataset_dir, model_path, int(train_iterations)),
                log_lines,
            ):
                yield basic_updates("Training model", model_path, log_lines, "running")

        if not any([run_convert, run_feature_compute, run_training]):
            log_lines.append("No setup stages selected.")
            trim_logs(log_lines)

        yield basic_updates("Setup stages completed", model_path, log_lines, "completed")
    except Exception as exc:
        log_lines.append(f"ERROR: {exc}")
        trim_logs(log_lines)
        try:
            resolved_model_path = derive_model_path(dataset_dir, model_dir) if str(dataset_dir).strip() else str(model_dir or "")
        except Exception:
            resolved_model_path = str(model_dir or "")
        yield basic_updates("Setup stages failed", resolved_model_path, log_lines, "failed")


def query_runner(
    run_heatmap,
    run_physics,
    dataset_dir,
    model_dir,
    heatmap_camera_slerp_list,
    heatmap_step_size,
    text_query,
    neg_text_query,
    fg_obj_list,
    bg_obj_list,
    ground_plane_name,
    rigid_object_name,
    threshold,
    object_select_eps,
    inward_bbox_offset,
    final_noise_filtering,
    rigid_speed,
    use_rigidity,
    final_camera_slerp_list,
    final_step_size,
):
    log_lines = []
    heatmap_gif = gr.skip()
    heatmap_gallery_paths = gr.skip()
    final_gif = gr.skip()
    final_gallery_paths = gr.skip()

    try:
        dataset_dir = str(dataset_dir).strip()
        if not dataset_dir:
            raise ValueError("dataset_dir is required.")
        if run_heatmap and not str(text_query).strip():
            raise ValueError("text_query is required when heatmap rendering is enabled.")
        if run_physics:
            if not str(fg_obj_list).strip():
                raise ValueError("fg_obj_list is required when physics is enabled.")
            if not str(bg_obj_list).strip():
                raise ValueError("bg_obj_list is required when physics is enabled.")
            if not str(ground_plane_name).strip():
                raise ValueError("ground_plane_name is required when physics is enabled.")

        model_path = derive_model_path(dataset_dir, model_dir)
        log_lines.append("Query-stage request accepted.")
        log_lines.append(f"Resolved model path: {model_path}")
        trim_logs(log_lines)
        yield query_updates("Preparing query stages", model_path, log_lines, heatmap_gif, heatmap_gallery_paths, final_gif, final_gallery_paths, "running")

        if run_heatmap:
            heatmap_render_started_at = time.time()
            for _ in run_command_stream(
                "Render heatmap",
                build_heatmap_command(
                    dataset_dir,
                    model_path,
                    heatmap_camera_slerp_list,
                    int(heatmap_step_size),
                    text_query,
                    neg_text_query,
                ),
                log_lines,
            ):
                yield query_updates("Rendering heatmap", model_path, log_lines, heatmap_gif, heatmap_gallery_paths, final_gif, final_gallery_paths, "running")

            heatmap_gif, heatmap_gallery_paths = collect_heatmap_outputs(model_path, heatmap_render_started_at, log_lines)
            yield query_updates("Heatmap ready", model_path, log_lines, heatmap_gif, heatmap_gallery_paths, final_gif, final_gallery_paths, "running")

        if run_physics:
            for _ in run_command_stream(
                "Segment object",
                build_segment_command(
                    model_path,
                    fg_obj_list,
                    bg_obj_list,
                    ground_plane_name,
                    rigid_object_name,
                    threshold,
                    object_select_eps,
                    inward_bbox_offset,
                    final_noise_filtering,
                ),
                log_lines,
            ):
                yield query_updates("Segmenting object", model_path, log_lines, heatmap_gif, heatmap_gallery_paths, final_gif, final_gallery_paths, "running")

            for _ in run_command_stream(
                "Run physics",
                build_physics_command(model_path, rigid_speed, use_rigidity),
                log_lines,
            ):
                yield query_updates("Running physics simulation", model_path, log_lines, heatmap_gif, heatmap_gallery_paths, final_gif, final_gallery_paths, "running")

            final_render_started_at = time.time()
            for _ in run_command_stream(
                "Render final animation",
                build_final_render_command(
                    dataset_dir,
                    model_path,
                    final_camera_slerp_list,
                    int(final_step_size),
                ),
                log_lines,
            ):
                yield query_updates("Rendering final animation", model_path, log_lines, heatmap_gif, heatmap_gallery_paths, final_gif, final_gallery_paths, "running")

            final_gif, final_gallery_paths = collect_final_outputs(model_path, final_render_started_at, log_lines)
            yield query_updates("Final animation ready", model_path, log_lines, heatmap_gif, heatmap_gallery_paths, final_gif, final_gallery_paths, "running")

        if not any([run_heatmap, run_physics]):
            log_lines.append("No query stages selected.")
            trim_logs(log_lines)

        yield query_updates("Query stages completed", model_path, log_lines, heatmap_gif, heatmap_gallery_paths, final_gif, final_gallery_paths, "completed")
    except Exception as exc:
        log_lines.append(f"ERROR: {exc}")
        trim_logs(log_lines)
        try:
            resolved_model_path = derive_model_path(dataset_dir, model_dir) if str(dataset_dir).strip() else str(model_dir or "")
        except Exception:
            resolved_model_path = str(model_dir or "")
        yield query_updates("Query stages failed", resolved_model_path, log_lines, heatmap_gif, heatmap_gallery_paths, final_gif, final_gallery_paths, "failed")


def launch_ui():
    with gr.Blocks(
        title="Feature Splatting Pipeline",
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="amber",
            neutral_hue="slate",
        ),
    ) as demo:
        gr.Markdown(
            """
            # Feature Splatting Pipeline
            Run the pipeline from a dataset directory: setup stages, an optional LLM-assisted image analysis step, then query-driven heatmap, segmentation, physics, and final rendering.
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                with gr.Group():
                    gr.Markdown("### Inputs")
                    with gr.Row():
                        dataset_dir = gr.Textbox(label="Dataset Directory", placeholder="feat_data/garden_table", value="feat_data/garden_table", scale=5)
                        browse_dataset_button = gr.Button("Browse Folder", scale=1)
                    model_dir = gr.Textbox(
                        label="Model Directory (optional)",
                        placeholder="Leave blank to use output/<dataset_name>",
                        value="",
                    )

                with gr.Group():
                    gr.Markdown("### Setup Stages (Before Text Query)")
                    with gr.Row():
                        run_convert = gr.Checkbox(label="Run COLMAP Convert", value=False)
                        convert_skip_matching = gr.Checkbox(label="Convert Skip Matching", value=False)
                        convert_resize = gr.Checkbox(label="Convert Resize Images", value=False)
                    with gr.Row():
                        run_feature_compute = gr.Checkbox(label="Run Feature Compute", value=True)
                        run_training = gr.Checkbox(label="Run Training", value=True)

                with gr.Group():
                    gr.Markdown("### Training (Setup)")
                    train_iterations = gr.Number(label="Train Iterations", value=10000, precision=0)

                setup_button = gr.Button("Run Setup Stages", variant="primary", size="lg")

                with gr.Group():
                    gr.Markdown("### LLM Assist (Between Setup and Query)")
                    gr.Markdown(
                        "Pick one reference image, ask the model for candidate foreground object groups, then select one to auto-fill the query form."
                    )
                    openai_model = gr.Textbox(label="OpenAI Model", value="gpt-5.2")
                    openai_api_key = gr.Textbox(
                        label="OpenAI API Key (optional; falls back to OPENAI_API_KEY)",
                        type="password",
                        value="",
                        placeholder="Use a fresh key; do not reuse an exposed key.",
                    )
                    llm_status_md = gr.Markdown("### LLM Assist Idle")
                    llm_raw_output = gr.Textbox(label="LLM Raw Output", lines=8, max_lines=12, interactive=False)
                    with gr.Row():
                        reference_image_path = gr.Textbox(
                            label="Reference Image",
                            placeholder="Select one image from the dataset",
                            value="",
                            scale=5,
                        )
                        browse_reference_image_button = gr.Button("Browse Image", scale=1)
                        suggest_foreground_button = gr.Button("Suggest Foreground Options")
                    reference_image_preview = gr.Image(label="Reference Image Preview", type="filepath")
                    selected_foreground_candidate = gr.Dropdown(
                        label="Candidate Foreground Option",
                        choices=[],
                        value=None,
                    )
                    autofill_button = gr.Button("Auto-Fill Query Fields", variant="primary")

                with gr.Group():
                    gr.Markdown("### Query Stages (After Text Query)")
                    with gr.Row():
                        run_heatmap = gr.Checkbox(label="Run Heatmap Render", value=True)
                        run_physics = gr.Checkbox(label="Run Segmentation + Physics + Final Render", value=True)
                    text_query = gr.Textbox(label="Text Query", value="a vase with flowers")
                    neg_text_query = gr.Textbox(label="Negative Text Query", value="")
                    with gr.Row():
                        heatmap_camera_slerp_list = gr.Textbox(label="Heatmap Camera Slerp List", value="0 1")
                        heatmap_step_size = gr.Number(label="Heatmap Step Size", value=10, precision=0)

                with gr.Group():
                    gr.Markdown("### Segmentation + Physics")
                    fg_obj_list = gr.Textbox(label="Foreground Objects", value="vase,flowers,plants")
                    bg_obj_list = gr.Textbox(label="Background Objects", value="tabletop,wooden table")
                    with gr.Row():
                        ground_plane_name = gr.Textbox(label="Ground Plane", value="tabletop")
                        rigid_object_name = gr.Textbox(label="Rigid Object", value="vase")
                    with gr.Row():
                        threshold = gr.Number(label="Threshold", value=0.6)
                        object_select_eps = gr.Number(label="Object Select EPS", value=0.1)
                        inward_bbox_offset = gr.Number(label="Inward BBox Offset", value=0.15)
                    with gr.Row():
                        final_noise_filtering = gr.Checkbox(label="Final Noise Filtering", value=True)
                        use_rigidity = gr.Checkbox(label="Use Rigidity", value=True)
                        rigid_speed = gr.Number(label="Rigid Speed", value=0.3)
                    with gr.Row():
                        final_camera_slerp_list = gr.Textbox(label="Final Camera Slerp List", value="54 58")
                        final_step_size = gr.Number(label="Final Step Size", value=500, precision=0)

                query_button = gr.Button("Run Query Stages", variant="primary", size="lg")

            with gr.Column(scale=6):
                status_md = gr.Markdown("### Idle")
                resolved_model_path = gr.Textbox(label="Resolved Model Path", interactive=False)
                logs_box = gr.Textbox(label="Logs", lines=22, max_lines=22, interactive=False)
                with gr.Row():
                    heatmap_preview = gr.Image(label="Heatmap Preview (GIF loops in browser)", type="filepath")
                    final_preview = gr.Image(label="Final Preview (GIF loops in browser)", type="filepath")
                with gr.Row():
                    heatmap_gallery = gr.Gallery(label="Heatmap Frames", columns=4, height=300)
                    final_gallery = gr.Gallery(label="Final RGB Frames", columns=4, height=300)

        setup_inputs = [
            run_convert,
            convert_skip_matching,
            convert_resize,
            run_feature_compute,
            run_training,
            dataset_dir,
            model_dir,
            train_iterations,
        ]
        query_inputs = [
            run_heatmap,
            run_physics,
            dataset_dir,
            model_dir,
            heatmap_camera_slerp_list,
            heatmap_step_size,
            text_query,
            neg_text_query,
            fg_obj_list,
            bg_obj_list,
            ground_plane_name,
            rigid_object_name,
            threshold,
            object_select_eps,
            inward_bbox_offset,
            final_noise_filtering,
            rigid_speed,
            use_rigidity,
            final_camera_slerp_list,
            final_step_size,
        ]
        setup_outputs = [
            status_md,
            resolved_model_path,
            logs_box,
        ]
        query_outputs = [
            status_md,
            resolved_model_path,
            logs_box,
            heatmap_preview,
            heatmap_gallery,
            final_preview,
            final_gallery,
        ]

        setup_button.click(
            fn=setup_runner,
            inputs=setup_inputs,
            outputs=setup_outputs,
        )

        browse_dataset_button.click(
            fn=browse_dataset_directory,
            inputs=[dataset_dir],
            outputs=[dataset_dir],
        )

        browse_reference_image_button.click(
            fn=browse_reference_image,
            inputs=[dataset_dir, reference_image_path],
            outputs=[reference_image_path, reference_image_preview],
        )

        reference_image_path.change(
            fn=preview_selected_reference_image,
            inputs=[reference_image_path],
            outputs=[reference_image_preview],
        )

        suggest_foreground_button.click(
            fn=suggest_foreground_candidates,
            inputs=[openai_api_key, openai_model, reference_image_path],
            outputs=[llm_status_md, llm_raw_output, selected_foreground_candidate],
        )

        autofill_button.click(
            fn=autofill_query_fields,
            inputs=[openai_api_key, openai_model, reference_image_path, selected_foreground_candidate],
            outputs=[llm_status_md, llm_raw_output, text_query, fg_obj_list, bg_obj_list, ground_plane_name, rigid_object_name],
        )

        query_button.click(
            fn=query_runner,
            inputs=query_inputs,
            outputs=query_outputs,
        )

    host = os.getenv("FEATURE_SPLAT_UI_HOST", "127.0.0.1")
    port = int(os.getenv("FEATURE_SPLAT_UI_PORT", "7860"))
    demo.queue().launch(server_name=host, server_port=port, show_error=True)


if __name__ == "__main__":
    launch_ui()
