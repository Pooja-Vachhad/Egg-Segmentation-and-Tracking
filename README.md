# Egg Segmentation and Tracking (CV Intern Task)

- Report: `report.pdf` (in this repo)
- Tracking video and all outputs (Google Drive): https://drive.google.com/file/d/1AQb4GCPZRGx-FoZV2DSINwa-Bl6W4fvk/view?usp=sharing

## What this repo does
| Task | What I did |
|------|-----------|
| 1 | Segment eggs with SAM (pretrained model) plus size, border and shadow filters |
| 2 | Split data 4:1 (dev/eval) and build a classical algorithm (final: algo0, brightness threshold) |
| 3 | Compare algo0 with the SAM masks (IoU, Dice, precision, recall) |
| 4 | Track eggs in a video with DeepSORT |

## Files
| File | Purpose |
|------|---------|
| `sam.py` | Task 1: makes SAM masks in `sam_masks/` |
| `split.py` | Task 2: makes `split.csv` (dev/eval) |
| `experiments.py` | Task 2: helper functions (fixed, Otsu, adaptive threshold, border removal). Not run directly. |
| `main.py` | Task 2: demo of algo0 on one image |
| `metrics.py` | Task 3: runs algo0 on all images, saves metrics and best/worst figures |
| `track.py` | Task 4: tracking on the video |
| `report.pdf` | Full report |

## Data (in the Drive folder)
- `folder/`: original egg images
- `sam_masks/`: SAM masks
- `bad_mask/`: images with wrong SAM masks (left out of the metrics)
- `split.csv`: dev/eval split
- `results/`: results of the three algorithms (overall metrics, per-image metrics, 5 best/worst images)
- `output.mp4`: tracking video

## Setup
1. Clone the repo and install the requirements:
```
   git clone <your repo link>
   cd <repo name>
   pip install -r requirements.txt
```
2. Download the SAM checkpoint (`vit_h`) and put it in the repo folder:
   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
3. Get the data. Choose one:
   - **Quick:** download `folder/`, `sam_masks/`, `bad_mask/` and `split.csv` from Drive and put them in the repo folder.
   - **From scratch:** put the original images in `folder/` and run `sam.py`.

A GPU is recommended for SAM. It also runs on CPU, but it is slow.

## How to run (in this order)
```
python sam.py       # SAM masks -> sam_masks/
python split.py     # dev/eval split -> split.csv
python metrics.py   # algo0 + metrics -> results/
python track.py     # tracking on the video
```
- For tracking: put the video in the same folder as `track.py`, and set its file name at the top of `track.py`:
```python
  VIDEO_PATH = "your_video.mp4"
```
- To try algo0 on a single image: put `egg.jpg` in the repo folder and run `python main.py`.

## Results files
- `results/algo0_overall.csv`: mean metrics for dev and eval
- `results/algo0_per_image.csv`: metrics for each image
- `results/best.png`, `results/worst.png`: best and worst examples

## Notes
- The 19 bad SAM masks were found by checking by eye and are left out of all metrics.
- Parameters were compared by eye on sample images, not with a metric.
- Parameters were chosen on dev only. Eval was run once with the final settings.

## AI tools used
- I used Claude to learn how SAM and segmentation work, and I also read blogs and articles.
- AI helped write the code that saves the CSV files and images.
- I ran all experiments and checked the results myself.