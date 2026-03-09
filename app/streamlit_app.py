"""Small Streamlit app for browsing TraceMap results."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from tracemap import TraceMap, TraceMapConfig
from tracemap.data import build_default_pet_datasets
from tracemap.visualization import overlay_heatmap, tensor_to_uint8_image

st.set_page_config(page_title="TraceMap", layout="wide")


def display_image(image_tensor, heatmap_tensor):
    """Turn an image and heatmap into something Streamlit can display."""
    overlay = overlay_heatmap(image_tensor, heatmap_tensor)
    return tensor_to_uint8_image(overlay).cpu().numpy()


@st.cache_resource(show_spinner=False)
def load_pipeline_and_data():
    """Load the cached model and pet split, or build them once."""
    config = TraceMapConfig()
    datasets = build_default_pet_datasets(config, download=True)
    cache_path = Path(config.cache_path)

    if cache_path.exists():
        pipeline = TraceMap.load_bundle(
            cache_path,
            config=config,
            train_dataset=datasets.train,
        )
    else:
        pipeline = TraceMap(config)
        pipeline.fit(datasets.train, datasets.val)
        pipeline.build_index(datasets.train)
        pipeline.save_bundle(cache_path)

    return pipeline, datasets


def render_example_list(result, examples, label):
    """Draw one result section."""
    st.subheader(label)
    st.caption("Lower scores support the chosen class. Higher scores push against it.")
    for rank, example in enumerate(examples, start=1):
        st.markdown(
            (
                f"**#{rank} {example.class_name}**"
                f"  | influence `{example.influence_score:+.4f}`"
                f"  | affinity `{example.affinity_score:.3f}`"
            ),
        )
        left, right = st.columns(2)
        left.image(
            display_image(result.query_image, example.query_heatmap),
            caption="Query evidence",
            use_container_width=True,
        )
        right.image(
            display_image(example.image, example.heatmap),
            caption=f"Training example #{example.dataset_index}",
            use_container_width=True,
        )
        if example.image_path:
            st.caption(example.image_path)


def main() -> None:
    """Run the app."""
    st.title("TraceMap")
    st.write(
        "See which training images pulled a prediction around, and which parts"
        " of the images seem to matter in each comparison.",
    )

    with st.spinner("Loading the model and pet split..."):
        pipeline, datasets = load_pipeline_and_data()

    st.sidebar.header("Query")
    source = st.sidebar.radio("Source", ("Test split", "Upload"))
    top_k = st.sidebar.slider("Top-k examples", min_value=1, max_value=8, value=5)
    selected_target = st.sidebar.selectbox(
        "Explain toward class",
        options=["Predicted class", *pipeline.config.class_names],
    )
    target = None if selected_target == "Predicted class" else selected_target

    query_input: Image.Image | object
    if source == "Test split":
        query_index = st.sidebar.number_input(
            "Test image index",
            min_value=0,
            max_value=max(len(datasets.test) - 1, 0),
            value=0,
            step=1,
        )
        raw_query = datasets.test.get_raw_image(int(query_index))
        query_input, _ = datasets.test[int(query_index)]
    else:
        uploaded = st.sidebar.file_uploader(
            "Upload an RGB image",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded is None:
            st.info("Upload an image to get started.")
            return
        raw_query = Image.open(uploaded).convert("RGB")
        query_input = raw_query

    with st.spinner("Working on the explanation..."):
        result = pipeline.explain(query_input, target=target, top_k=top_k)

    raw_col, overlay_col = st.columns(2)
    raw_col.subheader("Query image")
    raw_col.image(raw_query, use_container_width=True)
    overlay_col.subheader("Query heatmap")
    overlay_col.image(
        display_image(result.query_image, result.query_heatmap),
        use_container_width=True,
    )

    st.metric("Predicted class", result.prediction.class_name)
    st.metric("Confidence", f"{result.prediction.confidence:.3f}")

    helpful_tab, harmful_tab = st.tabs(["Helpful", "Harmful"])
    with helpful_tab:
        render_example_list(result, result.helpful_examples, "Helpful examples")
    with harmful_tab:
        render_example_list(result, result.harmful_examples, "Harmful examples")


if __name__ == "__main__":
    main()
