"""
Interactive viewer for uniform_clip_kmeans frame clustering.

This tool is separate from app.py and is intended for debugging/analysis.

Run:
  python3 tools/cluster_viz.py \
    --video data/shrek.mp4 \
    --start_sec 120 \
    --end_sec 150
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import plotly.graph_objects as go

# Ensure project root is importable when running as:
#   python tools/cluster_viz.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scene_detect import inspect_scene_clusters


def _frame_caption(frame: dict) -> str:
    mark = "SELECTED" if frame["selected"] else "discarded"
    return (
        f"idx={frame['index']} | t={frame['timestamp_sec']:.3f}s | "
        f"cluster={frame['cluster_id']} | {mark}"
    )


def _to_gallery_items(frames: list[dict]) -> list[tuple[str, str]]:
    return [(frame["image_path"], _frame_caption(frame)) for frame in frames]


def _to_table_rows(frames: list[dict]) -> list[list[Any]]:
    return [
        [
            frame["index"],
            frame["timestamp_sec"],
            frame["cluster_id"],
            "yes" if frame["selected"] else "no",
            frame["image_path"],
        ]
        for frame in frames
    ]


def _build_cluster_plot(frames: list[dict]) -> tuple[Any, dict[tuple[int, int], dict]]:
    if not frames:
        return go.Figure(), {}

    selected = [f for f in frames if f["selected"]]
    discarded = [f for f in frames if not f["selected"]]

    fig = go.Figure()
    point_map: dict[tuple[int, int], dict] = {}

    def add_group(
        items: list[dict],
        name: str,
        color: str,
        symbol: str,
        size: int,
        curve_number: int,
    ) -> None:
        if not items:
            return
        hover = []
        for point_number, f in enumerate(items):
            hover.append(
                (
                    f"<b>{name}</b><br>"
                    f"idx={f['index']} | t={f['timestamp_sec']:.3f}s | "
                    f"cluster={f['cluster_id']}<br>"
                    f"Click the point to preview frame below."
                )
            )
            point_map[(curve_number, point_number)] = f

        fig.add_trace(
            go.Scatter(
                x=[f["timestamp_sec"] for f in items],
                y=[f["cluster_id"] for f in items],
                mode="markers",
                name=name,
                marker={
                    "size": size,
                    "color": color,
                    "symbol": symbol,
                    "line": {"width": 1, "color": "#101828"},
                },
                hovertext=hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    add_group(discarded, "Discarded", "#64748b", "circle", 9, curve_number=0)
    add_group(selected, "Selected", "#2563eb", "diamond", 13, curve_number=1)

    fig.update_layout(
        title="Cluster map (x=time_sec, y=cluster_id). Hover a point for frame preview.",
        xaxis_title="timestamp_sec",
        yaxis_title="cluster_id",
        template="plotly_white",
        height=460,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        hoverlabel={"align": "left"},
    )
    return fig, point_map


def _build_summary(result: dict) -> str:
    return (
        f"video: `{result['video_path']}`\n\n"
        f"range: `{result['start_sec']:.3f}s - {result['end_sec']:.3f}s`\n\n"
        f"sample_fps: `{result['sample_fps']}`\n\n"
        f"candidates: `{result['n_candidates']}`\n\n"
        f"requested keyframes: `{result['requested_keyframes']}`\n\n"
        f"selected (n_select): `{result['n_selected']}`\n\n"
        f"selected indices: `{result['selected_indices']}`\n\n"
        f"saved debug dir: `{result['output_dir']}`"
    )


def run_inspection(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_fps: float,
    keyframes_per_scene: int,
    cluster_model: str,
    cluster_device: str,
    cluster_batch: int,
    cluster_seed: int,
    output_dir: str,
) -> tuple[str, list[list[Any]], list[tuple[str, str]], Any, dict, list[tuple[str, str]], dict]:
    result = inspect_scene_clusters(
        video_path=video_path,
        start_sec=start_sec,
        end_sec=end_sec,
        output_dir=output_dir,
        sample_fps=sample_fps,
        keyframes_per_scene=keyframes_per_scene,
        cluster_model=cluster_model,
        cluster_device=cluster_device,
        cluster_batch=cluster_batch,
        cluster_seed=cluster_seed,
    )
    frames = result["frames"]
    selected_frames = [f for f in frames if f["selected"]]
    cluster_choices = ["all"] + [str(cid) for cid in result["cluster_ids"]]
    cluster_plot, point_map = _build_cluster_plot(frames)
    result["point_map"] = {
        f"{curve}:{point}": frame
        for (curve, point), frame in point_map.items()
    }

    return (
        _build_summary(result),
        _to_table_rows(frames),
        _to_gallery_items(selected_frames),
        cluster_plot,
        gr.update(choices=cluster_choices, value="all"),
        _to_gallery_items(frames),
        result,
    )


def update_cluster_gallery(cluster_value: str, state: dict) -> list[tuple[str, str]]:
    frames = state.get("frames", []) if isinstance(state, dict) else []
    if not frames:
        return []
    if cluster_value == "all":
        filtered = frames
    else:
        cluster_id = int(cluster_value)
        filtered = [f for f in frames if int(f["cluster_id"]) == cluster_id]
    return _to_gallery_items(filtered)


def preview_from_plot(state: dict, evt: gr.SelectData):
    if not isinstance(state, dict) or not state:
        return None, "No state"
    if evt is None:
        return None, "Click a point to preview frame"

    payload = evt.value if isinstance(evt.value, dict) else {}
    curve = payload.get("curve_number")
    point = payload.get("point_number")
    if curve is None or point is None:
        # Fallback for some plotly/gradio versions.
        curve = getattr(evt, "curve_number", None)
        point = getattr(evt, "point_number", None)

    if curve is None or point is None:
        return None, f"Unsupported event payload: {evt.value!r}"

    key = f"{int(curve)}:{int(point)}"
    frame = (state.get("point_map") or {}).get(key)
    if not frame:
        return None, f"Point mapping not found for {key}"

    meta = (
        f"idx={frame['index']} | t={frame['timestamp_sec']:.3f}s | "
        f"cluster={frame['cluster_id']} | "
        f"{'SELECTED' if frame['selected'] else 'discarded'}"
    )
    return frame["image_path"], meta


def preview_from_table(state: dict, evt: gr.SelectData):
    frames = state.get("frames", []) if isinstance(state, dict) else []
    if evt is None:
        return None, "Select a table row to preview frame"
    row_index = getattr(evt, "index", None)
    if isinstance(row_index, (tuple, list)):
        row_index = row_index[0] if row_index else None
    if row_index is None:
        return None, "Could not read selected row index"
    try:
        row_index = int(row_index)
    except Exception:
        return None, f"Invalid row index: {row_index!r}"
    if row_index < 0 or row_index >= len(frames):
        return None, f"Row index out of range: {row_index}"

    frame = frames[row_index]
    meta = (
        f"idx={frame['index']} | t={frame['timestamp_sec']:.3f}s | "
        f"cluster={frame['cluster_id']} | "
        f"{'SELECTED' if frame['selected'] else 'discarded'}"
    )
    return frame["image_path"], meta


def build_demo(defaults: argparse.Namespace) -> gr.Blocks:
    with gr.Blocks(title="Cluster Inspector") as demo:
        gr.Markdown("## uniform_clip_kmeans cluster inspector")
        gr.Markdown(
            "Inspect candidate frames, clusters, and selected representative frames "
            "for one scene range."
        )

        state = gr.State({})

        with gr.Row():
            video_path = gr.Textbox(label="video_path", value=defaults.video)
            output_dir = gr.Textbox(
                label="debug_output_dir",
                value=defaults.output_dir,
            )

        with gr.Row():
            start_sec = gr.Number(label="start_sec", value=defaults.start_sec, precision=3)
            end_sec = gr.Number(label="end_sec", value=defaults.end_sec, precision=3)
            sample_fps = gr.Number(label="sample_fps", value=defaults.sample_fps, precision=3)
            keyframes_per_scene = gr.Slider(
                label="keyframes_per_scene (n_select target)",
                minimum=1,
                maximum=20,
                step=1,
                value=defaults.keyframes_per_scene,
            )

        with gr.Row():
            cluster_model = gr.Textbox(label="cluster_model", value=defaults.cluster_model)
            cluster_device = gr.Dropdown(
                label="cluster_device",
                choices=["cpu", "cuda", "mps"],
                value=defaults.cluster_device,
            )
            cluster_batch = gr.Slider(
                label="cluster_batch",
                minimum=1,
                maximum=128,
                step=1,
                value=defaults.cluster_batch,
            )
            cluster_seed = gr.Number(label="cluster_seed", value=defaults.cluster_seed, precision=0)

        run_btn = gr.Button("Run clustering", variant="primary")
        summary = gr.Markdown()

        frame_table = gr.Dataframe(
            headers=["idx", "timestamp_sec", "cluster_id", "selected", "image_path"],
            label="All candidate frames",
            interactive=False,
        )
        selected_gallery = gr.Gallery(
            label="Selected representative frames",
            columns=3,
            height=260,
            show_label=True,
            preview=True,
        )
        cluster_plot = gr.Plot(label="Interactive cluster plot")
        preview_meta = gr.Textbox(label="Point preview metadata", interactive=False)
        preview_image = gr.Image(label="Point preview frame", type="filepath", height=260)

        cluster_dropdown = gr.Dropdown(label="Cluster to inspect", choices=["all"], value="all")
        cluster_gallery = gr.Gallery(
            label="Frames in selected cluster",
            columns=4,
            height=420,
            show_label=True,
            preview=True,
        )

        run_btn.click(
            fn=run_inspection,
            inputs=[
                video_path,
                start_sec,
                end_sec,
                sample_fps,
                keyframes_per_scene,
                cluster_model,
                cluster_device,
                cluster_batch,
                cluster_seed,
                output_dir,
            ],
            outputs=[
                summary,
                frame_table,
                selected_gallery,
                cluster_plot,
                cluster_dropdown,
                cluster_gallery,
                state,
            ],
        )

        cluster_dropdown.change(
            fn=update_cluster_gallery,
            inputs=[cluster_dropdown, state],
            outputs=[cluster_gallery],
        )

        if hasattr(cluster_plot, "select"):
            cluster_plot.select(
                fn=preview_from_plot,
                inputs=[state],
                outputs=[preview_image, preview_meta],
            )

        if hasattr(frame_table, "select"):
            frame_table.select(
                fn=preview_from_table,
                inputs=[state],
                outputs=[preview_image, preview_meta],
            )

    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive viewer for uniform_clip_kmeans clusters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", default="data/shrek.mp4", help="Path to video file")
    p.add_argument("--start_sec", type=float, default=0.0)
    p.add_argument("--end_sec", type=float, default=30.0)
    p.add_argument("--sample_fps", type=float, default=1.0)
    p.add_argument("--keyframes_per_scene", type=int, default=3)
    p.add_argument("--cluster_model", default="openai/clip-vit-base-patch32")
    p.add_argument("--cluster_device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--cluster_batch", type=int, default=16)
    p.add_argument("--cluster_seed", type=int, default=42)
    p.add_argument("--output_dir", default="output/cluster_debug")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7862)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_demo(args)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
