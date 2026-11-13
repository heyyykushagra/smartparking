# 🅿️ Car Parking Occupancy Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A parking-space occupancy detector with a web dashboard and hourly analytics.
Mark the spaces in a parking lot once, then see which spaces are free (green) and
occupied (red) in an image or video. Log each reading, and find the busiest and
quietest hours of the day.

**Maintainer:** Kushagra Singh · [Your College] · GitHub [@heyyykushagra](https://github.com/heyyykushagra)

> **Based on:** [Car-Parking-Detection](https://github.com/8harath/Car-Parking-Detection),
> originally created by **Bharath K** and released under the MIT License.
> This fork keeps his detection pipeline and desktop tools, and adds the features listed below.

---

## ✨ What I added

| Feature | Where |
|---|---|
| **Streamlit dashboard** with Live, Hourly trends, Video and Layout tabs | `dashboard.py`, `.streamlit/config.toml` |
| **Hourly analytics**: readings, average/peak occupancy and free spaces per hour, busiest and quietest hour, PNG chart, and a CLI | `occupancy_analytics.py` |
| **Headless core**: the same CV pipeline without GUI windows or YOLO, so it runs in Streamlit and in tests | `parking_core.py` |
| **Position-file loader fix**: `CarParkPos` can be a pickle (what the GUI saves) *or* plain-text `x,y` lines. The shipped file was text, and the old `pickle.load` couldn't read it | `parking_core.load_positions`, used by `enhanced_parking_detector.py` and `main.py` |
| **Video resolution scaling**: `carPark.mp4` is ~1.45× the size of `carParkImg.jpg`. Positions and box size are scaled by *s*, and the pixel threshold by *s²*, when the aspect ratios match | `parking_core.scale_factor` / `analyze_frame` |
| **Grid layout builder**: build rows/columns of spaces with a live preview, saved as a GUI-compatible pickle | dashboard → *Layout* tab |
| **Sample data generator**: a **synthetic** office-lot pattern written to a separate file | `generate_sample_data.py` |
| **Tests**: synthetic images and temporary CSVs, with no GUI and no YOLO. Tests can no longer overwrite the real `CarParkPos` | `tests/test_parking_core.py`, `tests/test_occupancy_analytics.py`, `tests/conftest.py` |

---

## 🚀 Quick start

```powershell
# Windows / PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(macOS/Linux: `source venv/bin/activate`.)

### 1. Dashboard

```powershell
streamlit run dashboard.py
```

- **Live**: analyse the sample image or one you upload. Shows total, occupied and free spaces and occupancy %. *Save this reading to log* appends a row to `data/parking_status.csv`.
- **Hourly trends**: pick a day and see an hourly bar chart with the peak marked, the busiest and quietest hour, a summary table, and a CSV download.
- **Video**: sample `carPark.mp4` (or an uploaded video) every N seconds. To log the samples with real timestamps, set *Recording started at…*.
- **Layout**: view the current spaces, or build a grid of spaces and save it to `CarParkPos`.

Use the sidebar to choose the image source, the occupancy threshold, and which log to read (real or sample).

### 2. Hourly analytics (CLI)

```powershell
python occupancy_analytics.py                       # real log
python occupancy_analytics.py --date 2026-11-13     # one day only
python occupancy_analytics.py --csv data/sample_parking_status.csv
```

Prints the hourly table and the busiest and quietest hour, and saves a chart to `reports/hourly_occupancy_*.png`.

### 3. Sample data (synthetic)

```powershell
python generate_sample_data.py          # 7 days, 15-minute readings
```

This writes **made-up demo data** to `data/sample_parking_status.csv`. It follows an office-lot
pattern: quiet nights, a morning rush, a lunch dip, an evening drain-out, and quieter weekends.
**It is not real detection output.** The script refuses to write to the real log, and the dashboard
labels it as sample data.

### 4. Original desktop tool

```powershell
python run.py --image carParkImg.jpg     # mark spaces (drag), S = save, D = detect + report
python run.py --video carPark.mp4        # live video view
```

Keys: `D` detect & report · `S` save layout · `R` reset · `Z` undo · right-click removes a space · `Q` quit.

---

## 📍 Marking parking spaces

The shipped `CarParkPos` is only a **starter layout** for `carParkImg.jpg`. For your own image
(or to improve the sample), re-mark the spaces using either:

- `python run.py --image your_image.jpg`, then drag to select a grid of spaces and press `S`, or
- the dashboard's **Layout** tab: set start x/y, columns, rows and steps, check the preview, then *Save layout*.

Both save the same pickle format. The dashboard keeps the previous file as `CarParkPos.bak`.

**About accuracy:** no accuracy figures have been measured for this project. Results depend mostly
on how well the spaces are marked (box position and size) and on the occupancy threshold, so
check the annotated image and adjust the threshold for your footage.

---

## 📒 Occupancy log format

`data/parking_status.csv` (falls back to `./parking_status.csv` if missing):

```
Total Slots,Occupied Slots,Available Slots,Timestamp
64,39,25,2026-11-13 09:15:00
```

The timestamp format is `%Y-%m-%d %H:%M:%S`. It is the same schema the original project used,
so old logs keep working.

---

## 🔍 How detection works

1. Grayscale → Gaussian blur → adaptive threshold → median blur → dilate
2. For each marked space, count the non-zero (edge) pixels
3. Count ≥ `OCCUPANCY_THRESHOLD` (default 900) → **occupied**, otherwise **free**
4. For frames larger than the image the spaces were marked on (same aspect ratio), positions and
   box size are scaled by *s* and the threshold by *s²*

The desktop tool (`run.py`, `D` key) also runs YOLOv8 vehicle detection for its reports. The
dashboard and `parking_core.py` don't import YOLO, so they stay lightweight.

Settings live in `config.py` (`PARKING_WIDTH`, `PARKING_HEIGHT`, `OCCUPANCY_THRESHOLD`, paths, colours).

---

## 📁 Project structure

```
dashboard.py               Streamlit dashboard (new)
parking_core.py            Headless detection core (new)
occupancy_analytics.py     Hourly analytics + CLI (new)
generate_sample_data.py    Synthetic sample data (new)
.streamlit/config.toml     Dashboard theme (new)
run.py / gui.py / main.py  Original desktop entry points
enhanced_parking_detector.py, car_detector.py, config.py
CarParkPos                 Parking-space layout (starter)
carParkImg.jpg, carPark.mp4  Sample media
data/                      Occupancy logs
reports/                   Generated charts and reports
tests/                     pytest suite
```

---

## 🧪 Tests

```powershell
python -m pytest
```

The new tests cover the core pipeline, the position loader, resolution scaling, video sampling,
log loading, hourly stats and the CLI. Three older tests in `test_parking_detector.py` /
`test_integration.py` were already failing before this fork: they expect a `zoom_level` attribute,
a pre-loaded `current_image`, and an exception for a missing image, none of which the detector has.

---

## 📄 License

MIT. See [LICENSE](LICENSE). The original copyright (Bharath K, 2024) is kept, and this fork's
modifications are added below it.

## 🙏 Credits

- **Bharath K**: original [Car-Parking-Detection](https://github.com/8harath/Car-Parking-Detection) project
- **[Ultralytics](https://github.com/ultralytics/ultralytics)**: YOLOv8
- **[Computer Vision Zone](https://www.computervision.zone/)** (Murtaza Hassan): the original parking-space counter idea and `cvzone`
- Libraries: OpenCV, NumPy, pandas, Matplotlib, Streamlit, Altair
