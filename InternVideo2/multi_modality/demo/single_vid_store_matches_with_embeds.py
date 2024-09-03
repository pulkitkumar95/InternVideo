"""
Really just a script I made to dump video summaries so I can use for hand evaluation of matches in a video
Use create_html to create the website
"""
import argparse
import clip
import cv2
import pandas as pd
import os
import hashlib
import json
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import shutil
from sklearn.metrics.pairwise import cosine_similarity
import torch

from matplotlib.animation import FuncAnimation

import sys
sys.path.append('/fs/cfar-projects/inr_analysis/pulkit/InternVideo/InternVideo2/multi_modality')
sys.path.append('/fs/cfar-projects/inr_analysis/pulkit/InternVideo/InternVideo2/multi_modality/demo')
sys.path.append('/fs/cfar-projects/inr_analysis/pulkit/InternVideo/InternVideo2/multi_modality/models')
# sys.path.append('/fs/cfar-projects/inr_analysis/pulkit/InternVideo/InternVideo2/multi_modality/dataset/')

from config import Config, eval_dict_leaf
from utils import setup_internvideo2, get_text_features as get_internvideo2s2_text_features

def load_embeds(file_path):
    if ".json" in file_path:
        with open(file_path, "r") as f:
            return json.load(f)
    elif ".pkl" in file_path:
        sys.path.append('/fs/cfar-projects/inr_analysis/pulkit/InternVideo/InternVideo2/multi_modality/dataset')
        with open(file_path, "rb") as f:
            return pickle.load(f)
    print("Loaded embeds")

def remove_repetitions(text):
    # Step 1: Use a regex to find repeated words or sequences
    # This regex matches any word that is followed by the same word multiple times
    pattern = re.compile(r"\b(\w+)( \1\b)+")

    # Step 2: Substitute the repeated sequences with a single occurrence of the word
    cleaned_text = pattern.sub(r"\1", text)

    return cleaned_text


def get_clip_text_embedding(text, model, device):
    with torch.no_grad():
        text_tokens = clip.tokenize([text], truncate=True).to(device)
        text_features = model.encode_text(text_tokens)
    return text_features.cpu().numpy()

def get_internvideo2s2_text_embedding(text, model, device):
    # Use the text encoder of InternVideo2_Stage2 to encode the input query
    with torch.no_grad():
        text_features = get_internvideo2s2_text_features(text, model, device)
    return text_features.cpu().numpy()

def save_hashed_queries(text_queries, video_id):
    hashed_queries = {}
    for query in text_queries:
        query_id = hashlib.md5(query.encode()).hexdigest()
        hashed_queries[query_id] = query
    os.makedirs(os.path.join(output_dir, video_id, "queries"), exist_ok=True)
    with open(
        os.path.join(output_dir, video_id, "queries", "hashed_queries.json"), "w"
    ) as f:
        json.dump(hashed_queries, f, indent=2)

    return hashed_queries


def save_video_info(video_id, avg_embeds, frame_captions, output_dir):
    video_dir = os.path.join(output_dir, video_id, "info")
    os.makedirs(video_dir, exist_ok=True)
    # Create a dictionary that maps frame indices to BLIP captions
    frame_to_caption = dict(
        zip(
            frame_captions[video_id]["orig_frame_indices"],
            frame_captions[video_id]["frame_captions"],
        )
    )
    subset_frame_captions = []

    # i) Save 8 evenly sampled frames
    video_data = avg_embeds[video_id]
    frame_indices = np.linspace(0, 63, 8, dtype=int)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, ax in enumerate(axes.flat):
        frame_idx = video_data["orig_frame_indices"][frame_indices[i]]
        frame_path = os.path.join(
            anet_1fps_path, f"{video_id}/frame_{(frame_idx+1):04d}.jpg"
        )
        frame = Image.open(frame_path)
        ax.imshow(frame)
        ax.axis("off")
        ax.set_title(f"Frame {frame_idx}")

        subset_frame_captions.append(remove_repetitions(frame_to_caption[frame_idx]))

    plt.tight_layout()
    plt.savefig(os.path.join(video_dir, "video_summary_2x4.png"))
    plt.close()

    # ii) Save GT captions and timestamps
    segment_timestamps = [
        segment["timestamp"]
        for _, segment in video_data["segments_info"]["segments"].items()
    ]
    segment_captions = [
        segment["sentence"].strip()
        for _, segment in video_data["segments_info"]["segments"].items()
    ]
    video_info = {
        "duration": video_data["segments_info"]["duration"],
        "segment_timestamps": segment_timestamps,
        "segment_captions": segment_captions,
        "activity_label": video_data["activity_label"],
        "orig_frame_indices": video_data["orig_frame_indices"].tolist(),
        "subset_frame_captions": subset_frame_captions,
    }
    with open(os.path.join(video_dir, "video_info.json"), "w") as f:
        json.dump(video_info, f, indent=2)


def display_results(video_id, results, video_data, blip_frame_captions, video_path):

    # Create a dictionary that maps frame indices to BLIP captions
    frame_to_caption = dict(
        zip(
            blip_frame_captions["orig_frame_indices"],
            blip_frame_captions["frame_captions"],
        )
    )

    for i, (similarity, result_type, data) in enumerate(results, 1):
        if result_type == "frame":
            frame_idx = video_data["orig_frame_indices"][data]
            print(f"\nMatch {i}: Frame {frame_idx+1} (Similarity: {similarity:.4f})")
            print(f"BLIP caption for frame: {frame_to_caption[frame_idx]}")
            frame_path = os.path.join(
                video_path, f"{video_id}/frame_{(frame_idx+1):04d}.jpg"
            )  # v___c8enCfzqw/frame_0001.jpg
            # print(frame_path)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(frame)
            plt.axis("off")
            plt.title(f"Frame {frame_idx}")
            plt.show()
        else:  # segment
            segment_idx, segment = data
            print(
                f"\nMatch {i}: Segment {segment_idx+1} (Similarity: {similarity:.4f})"
            )
            start_time, end_time = segment["timestamp"]
            print(
                f"Timestamp: {start_time:.2f}s - {end_time:.2f}s GT Caption: {segment['sentence']}"
            )

            frames = []
            for frame_idx in sorted(segment["orig_frame_indices"]):
                frame_path = os.path.join(
                    video_path, f"{video_id}/frame_{(frame_idx+1):04d}.jpg"
                )
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            fig, ax = plt.subplots()
            ax.axis("off")
            im = ax.imshow(frames[0])

            def update(frame):
                im.set_array(frame)
                return [im]

            # Create the animation and assign it to a variable to prevent it from being garbage collected
            anim = FuncAnimation(fig, update, frames=frames, blit=True, interval=100)
            plt.title(f"Segment: {start_time:.2f}s - {end_time:.2f}s")
            plt.show()


# def search_video(video_id, text_query, frame_seg_embeds, clip_model, device, embeds_type, top_k=5):
def search_video(video_id, text_query, frame_seg_embeds, text_model, device, embeds_type, top_k=5):
    video_data = frame_seg_embeds[video_id]
    # text_embed = get_clip_text_embedding(text_query, clip_model, device)
    if embeds_type == 'clip':
        text_embed = get_clip_text_embedding(text_query, text_model, device)
        frame_embeds_key = "frame_clip_embeds"
    elif embeds_type == 'internvideo2s2':
        text_embed = get_internvideo2s2_text_embedding(text_query, text_model, device)
        frame_embeds_key = "frame_embeds"

    frame_similarities = cosine_similarity(text_embed, video_data[frame_embeds_key])[
        0
    ]

    # For short videos <64 frames, discard duplicates in the top_k
    # Create a dictionary to store the highest similarity for each unique frame
    unique_frame_similarities = {}
    for idx, sim in enumerate(frame_similarities):
        orig_frame_idx = video_data["orig_frame_indices"][idx]
        if (
            orig_frame_idx not in unique_frame_similarities
            or sim > unique_frame_similarities[orig_frame_idx][0]
        ):
            unique_frame_similarities[orig_frame_idx] = (sim, idx)
    frame_results = [
        (float(sim), "frame", idx)
        for orig_idx, (sim, idx) in unique_frame_similarities.items()
    ]

    segment_similarities = []

    # TODO - change depending on final design
    segment_embeds_key = "segment_avg_embed" if embeds_type == "clip" else "segment_embeds"
    for segment_idx, segment in video_data["segments_info"]["segments"].items():
        if not np.sum(segment[segment_embeds_key]) == 0:
            # comparing a single text embedding with a single segment embedding
            similarity = cosine_similarity(text_embed, segment[segment_embeds_key])[
                0
            ][0]
            segment_similarities.append((float(similarity), "segment", (segment_idx, segment)))

    # frame_results = [(sim, 'frame', idx) for idx, sim in enumerate(frame_similarities)]
    all_results = frame_results + segment_similarities

    top_results = sorted(all_results, key=lambda x: x[0], reverse=True)[:top_k]

    # query_id = hashlib.md5(text_query.encode()).hexdigest()
    # return top_results, query_id
    return top_results


def save_query_results(
    video_id, query_id, query, results, frame_seg_embeds, frame_captions, output_dir
):
    # query_id = hashlib.md5(query.encode()).hexdigest()
    query_dir = os.path.join(output_dir, video_id, "queries", query_id)
    os.makedirs(os.path.join(query_dir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(query_dir, "segments"), exist_ok=True)

    video_data = frame_seg_embeds[video_id]
    frame_to_caption = dict(
        zip(
            frame_captions[video_id]["orig_frame_indices"],
            frame_captions[video_id]["frame_captions"],
        )
    )

    results_info = []

    for i, (similarity, result_type, data) in enumerate(results):
        if result_type == "frame":
            frame_idx = video_data["orig_frame_indices"][data]
            frame_path = os.path.join(
                anet_1fps_path, f"{video_id}/frame_{(frame_idx+1):04d}.jpg"
            )
            output_path = os.path.join(query_dir, "frames", f"match_{i}.png")
            shutil.copy(frame_path, output_path)

            results_info.append(
                {
                    "match_number": i,
                    "similarity_score": similarity,
                    "frame_number": int(frame_idx),
                    "type": "frame",
                    "image_path": f"frames/match_{i}.png",
                    "caption": remove_repetitions(frame_to_caption[frame_idx]),  # BLIP caption
                }
            )
            # breakpoint()
            
        else:
            segment_idx, segment = data
            start_time, end_time = segment["timestamp"]

            # Create gif for segment
            frames = []
            for frame_idx in sorted(segment["orig_frame_indices"]):
                frame_path = os.path.join(
                    anet_1fps_path, f"{video_id}/frame_{(frame_idx+1):04d}.jpg"
                )
                frames.append(Image.open(frame_path))

            output_path = os.path.join(query_dir, "segments", f"match_{i}.gif")
            gif_fps = 4
            gif_duration = 1000 // gif_fps
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=gif_duration,  # 100 #10*len(frames),
                loop=0,
            )

            results_info.append(
                {
                    "match_number": i,
                    "similarity_score": similarity,
                    "segment_number": int(segment_idx),
                    "type": "segment",
                    "gif_path": f"segments/match_{i}.gif",
                    "timestamp_range": [start_time, end_time],
                    "gt_caption": segment["sentence"],
                }
            )
            # breakpoint()

    # breakpoint()
    with open(os.path.join(query_dir, "results.json"), "w") as f:
        json.dump(results_info, f, indent=2)

def load_intern_model():
    # Load the InternVideo2_stage2
    print("Loading InternVideo2_stage2 model...")
    config = Config.from_file('/fs/cfar-projects/inr_analysis/pulkit/InternVideo/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py')
    config = eval_dict_leaf(config)
    model_pth = '/fs/cfar-projects/inr_analysis/semantic_inr/InternVideo2-stage2_1b-224p-f4.pt'
    config['pretrained_path'] = model_pth
    size_t = config.get('size_t', 224)
    
    intern_model, tokenizer = setup_internvideo2(config)

    return config, intern_model, tokenizer

def main(
    anet_1fps_path,
    val1_captions_file,
    val1_embeds_file,
    output_dir,
    embeds_type="clip",
    video_ids=None,
    text_queries_set=None,
    query_descriptiveness_set=None,
):

    if video_ids is None:
        # video_ids = ["v_a2HjLtnVDaY", "v_ROMy00dG8Ds"]  # , "v_9njq_aC4AS4"]
        video_ids = ["v_a2HjLtnVDaY", "v_ROMy00dG8Ds", "v_I9NukwdINyY", "v_f6Id4KERnoI"]
        text_queries_set = [
            [
                "A large dog is holding on to a blue leash that is on a smaller dog",
                "He stops ahead of the owner to wait for her to begin moving again",
                "Four stationary cars in the background behind the dogs",
                "a woman walking two dogs down the street",
                "Large white strip on a road lined by a green bush",
                "A large dog holds onto a leash on a small dog while the owner walks a third dog",
            ],
            [
                "The dog stops to sniff the wall as the cat continues on",
                "Both animals stop to look around.",
                "The cat stops by a yellow car in front of a house",
                "The cat reaches its paw out to touch the dog",
                "The cat and dog walk around dried fallen leaves on a damp brick sidewalk",
                "The cat looks at the camera",
            ],
            [
                "woman is standing in front of a table holding paintbrushes and talking to the camera about them.",
                "woman holds the brush and start painting on the white canvas.",
                "woman painting on a white canvas with a brush while holding another brush in the other hand.",
                "woman in a blue shirt in front view with art supplies.",
                "close up view of a canvas being painted upon.",
            ],
            [
                "A woman and a little girl are both using paint brushes on a piece of furniture.",
                "A woman and a little girl are painting the furniture white.",
                "A little girl painting a chair white alone.",
                "A green jug next to the base of a ladder.",
                "A can of white paint.",
                "Multiple cans of miscellaneous chemicals.",
            ],
        ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    val1_embeds = load_embeds(val1_embeds_file)
    val1_frame_captions = load_embeds(val1_captions_file)
    
    if embeds_type == "clip":
        text_model = clip_model
    elif embeds_type == "internvideo2s2":
        intern_config, text_model, intern_tokenizer = load_intern_model()
        

    for video_id, vid_text_queries in zip(video_ids, text_queries_set):#, query_descriptiveness_set):
        print(f"Processing video {video_id}...")

        # Save video info once
        save_video_info(video_id, val1_embeds, val1_frame_captions, output_dir)

        # Save hashed queries
        hashed_queries = save_hashed_queries(vid_text_queries, video_id)

        # Process each query
        for text_query in vid_text_queries:
            results = search_video(
                # video_id, text_query, val1_embeds, clip_model, device, embeds_type,
                video_id, text_query, val1_embeds, text_model, device, embeds_type,
            )

            # get hash encoded query_id from hashed_queries
            query_id = [k for k, v in hashed_queries.items() if v == text_query][0]

            save_query_results(
                video_id,
                query_id,
                text_query,
                results,
                val1_embeds,
                val1_frame_captions,
                output_dir,
            )
        # display_results(video_id, results, val1_avg_embeds[video_id], val1_frame_captions[video_id], anet_1fps_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeds_type", type=str, required=True)
    
    embeds_type = parser.parse_args().embeds_type
    
    # Two options: generate results for i) CLIP frame encoder + averaging 
    # ii) video encoder
    
    # Change text encoding accordingly. Also, change path to embeddings files
    # Also add the descriptiveness of the text queries to results
    
    anet_1fps_path = "/fs/cfar-projects/nerv-gan/10k/data_10k/activitynet/processed_videos/frames_fps1"
    val1_blip_captions_file = "/fs/cfar-projects/inr_analysis/semantic_inr/data/anet/val1_blip_captions.json"
    bucketed_queries_csv = "/fs/cfar-projects/inr_analysis/semantic_inr/data/anet/anet_avg_embeds/single_vid_clip_avg_matches.csv"

    if embeds_type == "clip":
        val1_embeds_file = "/fs/cfar-projects/inr_analysis/semantic_inr/data/anet/anet_avg_embeds/val1_avg_clip_embeds.pkl"
        output_dir = "/fs/cfar-projects/inr_analysis/semantic_inr/results/avg_nn_top5_results/"
        # bucketed_queries_csv = "/fs/cfar-projects/inr_analysis/semantic_inr/data/anet/anet_avg_embeds/single_vid_clip_avg_matches.csv"
    elif embeds_type == "internvideo2s2":
        val1_embeds_file = "/fs/cfar-projects/inr_analysis/semantic_inr/data/anet/anet_internvideo2s2_embeds/val_subset_embeds.pkl"
        output_dir = "/fs/cfar-projects/inr_analysis/semantic_inr/results/internvideo2s2_nn_top5_results/"
        # bucketed_queries_csv = "/fs/cfar-projects/inr_analysis/semantic_inr/data/anet/anet_internvideo2s2_embeds/single_vid_internvideo2s2_matches.csv"
    
    # Read the csv file and get the queries for each video_id
    # Also obtain the descriptiveness for each query
    df = pd.read_csv(bucketed_queries_csv)
    df = df.dropna(how="all") # Drop nans
    df = df[df['video_id']!='v__3xMhj4mbsk']
    video_ids = df["video_id"].unique().tolist()
    text_queries_set = []
    query_descriptiveness_set = []
    for video_id in video_ids:
        queries = df[df["video_id"] == video_id]["query"].tolist()
        text_queries_set.append(queries)
        query_descriptiveness = df[df["video_id"] == video_id]["verbosity_bucket"].tolist()
        query_descriptiveness_set.append(query_descriptiveness)

    # main(anet_1fps_path, val1_embeds_captions_file, val1_avg_embeds_file, output_dir)
    main(
        anet_1fps_path,
        val1_blip_captions_file,
        val1_embeds_file,
        output_dir,
        embeds_type=embeds_type,
        video_ids=video_ids,
        text_queries_set=text_queries_set,
        query_descriptiveness_set=query_descriptiveness_set,
    )
