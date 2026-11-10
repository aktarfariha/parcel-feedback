# Parcel Feedback: mapping land parcels in Bangladesh with SAM and human feedback

**Fariha Aktar** · Urban and Regional Planning, BUET · [Research portfolio](https://aktarfariha.github.io/) · Ongoing, started August 2026

> **Work in progress.** The pipeline and review tool are built and tested; labelling and evaluation
> on study areas in Bangladesh are under way, and results will be added here when they are ready.

<!-- Optional: a screenshot of the review page (the tool, not results) helps readers picture it.
     Upload it to figures/ and remove the arrows around the next line:
![The review page: SAM parcels over satellite imagery, ready for labelling](figures/review_page.png) -->

## Why this project

Up-to-date parcel boundaries are basic data for land-use planning, agricultural statistics and
flood or heat risk assessment, but in much of Bangladesh they are missing, outdated or locked in
paper cadastral maps. Fields are small, irregular and change between seasons, and ponds, homesteads
and tree patches sit between them. General-purpose segmentation models such as the Segment Anything
Model (SAM) can outline these parcels from high-resolution imagery without training data, but their
output still needs people to check it. This project tests how far SAM gets on its own, and builds a
simple way for people to correct and label its output so that a second model can learn from them.

## What the system does

1. **Detect.** The user draws a box on a satellite map. The image tiles (Esri World Imagery, about
   0.6 m per pixel at zoom 18) are stitched, georeferenced and sent to SAM (ViT-H, automatic mask generator)
   on a Colab GPU. Masks are turned into polygons, simplified, filtered by size, and overlapping
   masks are removed (smallest first).
2. **Review.** In a web map, people label each parcel (crop field, pond, house, road, tree cover,
   water, other), redraw wrong boundaries, and draw parcels SAM missed. A blind mode hides the
   model's guess and other people's answers so they cannot bias the labeller.
3. **Learn.** A Random Forest is trained on 35 measurements of colour, texture, shape and
   surroundings for each labelled parcel. It retrains automatically as labels arrive, and the
   "next parcel" button picks parcels where the model is least sure or labels that are still rare
   (active learning).
4. **Evaluate honestly.** Accuracy is reported only on parcels the model did not train on, and also
   on a whole area left out of training, with a learning curve showing accuracy against the number
   of labels.

## Status

- **Built and tested:** tile download and georeferencing, the SAM detector on a Colab GPU, the review
  page (labelling, boundary redrawing, adding missed parcels, GeoJSON export), and the learning loop
  with automatic retraining and held-out evaluation. Automated tests run on synthetic images.
- **Under way:** labelling parcels in study areas in Bangladesh and measuring how well SAM's
  boundaries and the label model hold up.
- **Results:** not reported yet; they will be added once evaluation is complete.

## Limitations and next steps

- SAM works on images of about 1024 px, so each run covers roughly 300 to 700 m; larger areas need tiling.
- Results depend on the season and date of the base imagery, which the tile service does not report.
- The 35 hand-made features could be replaced by image embeddings from a pre-trained model computed
  on the GPU, keeping the rest of the pipeline the same.
- Planned: compare the detected boundaries with an independent reference (for example survey or
  cadastral maps) and test how well the approach transfers across districts.

## Built with

Segment Anything (Kirillov et al., 2023) · FastAPI · Leaflet · scikit-learn · OpenCV · Shapely ·
Esri World Imagery (attribution shown in the app; check the provider's terms before publishing results).

---

# How to run it

```
 Your computer (web app)  --->  satellite tiles (Esri)
        |  sends the picture
        v
 Google Colab GPU (detector)  --->  returns parcel polygons  --->  review page
```

## A. Start the detector in Colab (every session)
1. Open https://colab.research.google.com -> **File -> Upload notebook** ->
   choose `colab/parcel_detector_colab.ipynb`.
2. **Runtime -> Change runtime type -> T4 GPU.**
3. **Runtime -> Run all.** The model download takes a few minutes the first time.
4. The last cell prints a **CONNECTION LINK**. Copy it. Leave that cell running.

The link changes every time you start the notebook.

## B. Start the web app on your computer
1. Clone or download this repository.
2. In a terminal inside this folder:
       pip install -r requirements.txt
       uvicorn app.main:app --reload
3. Open http://127.0.0.1:8000

## C. Use it
1. Click **Add new area**.
2. Paste the connection link and click **Connect** (green dot = ready).
3. Search a place (or type `lat, lon`), zoom in, click **Draw a box** and drag over your land
   (or **Use what I see**). The panel shows the image size and resolution.
4. Click **Detect parcels**. It downloads tiles, sends them to Colab, and opens the result
   in **Review parcels** when done (1 to 5 minutes on "Fast").
5. Label parcels, report boundary problems, draw missed parcels.
6. **Export GeoJSON** saves parcels + labels as a map file (lon/lat) for QGIS or further analysis.

## D. The learning part
The system learns from the labels people give. No internet service, no tokens and no GPU are needed.

- Click any parcel: the view **zooms in** on it (switch off with "Zoom in on click"; "Show whole image"
  zooms out) and the panel shows **the system's guess and how sure it is**, with the top three
  candidates. Answer with **Yes** (the system was right), **No** (then pick what it really is) or **Skip**.
  Keys: **Y** yes, **N** no, **S** skip, **1-7** pick a label. After each answer the next parcel opens
  by itself, so labelling is one key press per parcel.
- **Colour by** (top bar): people's agreement, the system's guess, or the system's confidence.
- **Retraining is automatic**: after every 5 new or changed answers it trains again in the background
  (a toast says "The system learned from new answers"). The first model appears once 12 parcels are
  labelled with at least 2 different labels. People-drawn missed parcels that carry a label are
  training examples too.
- **Next parcel** asks where the system is least sure, and more often about labels it has seen
  rarely (for example ponds), so the missing types fill up faster (active learning).
- **Blind mode** hides the system's guess and other people's answers until you answer, so they cannot
  bias you. Each answer also records what the system guessed at that moment, so "matches people in real
  use" is measured honestly.
- **Learning tab**: accuracy on parcels it did not study, accuracy on a brand-new area (leave one
  picture out), the learning curve (accuracy vs number of labels), per-label results, what it looks
  at, and the history of every retraining.

How it works: `app/learner.py` measures colour, texture, shape and surroundings for each parcel
(35 numbers), and trains a Random Forest on the consensus label of each labelled parcel. The model is
saved in `model/` (delete that folder to start over). Accuracy is always measured on held-back parcels.

Try it without your own data: `python tests/make_synthetic.py some_folder` makes stand-in pictures,
`python tests/test_learning.py some_folder` runs a full simulated test.

Possible upgrade later: replace the 35 hand-made measurements with picture embeddings from an open
model (for example CLIP) computed on the Colab GPU. The rest of the pipeline stays the same.

## E. Fixing a wrong boundary
Click the parcel, then **Redraw this boundary**. Click points around the TRUE edge, one after another,
then press **Finish** (Enter). The yellow dashed line shows the outline being replaced. Backspace undoes a
point, Esc cancels. The corrected outline replaces the old one on the map, in the GeoJSON export and in
the measurements the learner uses; the original outline is kept in the data, and **Restore original
outline** brings it back. The latest redraw is the one shown, whoever made it. Outlines that cross
themselves or are too small are refused with a message. A parcel reported as "not a parcel" is dimmed on
the map (not deleted).

Test it with stand-in pictures: `python tests/test_redraw.py some_folder`.

## Notes
- The selected area must fit in about 1024 px (what SAM works at). Large boxes get a lower
  zoom (blurrier); small boxes give sharper parcels. The best detail is a box of 300-700 m.
- Satellite imagery: Esri World Imagery by default, with its attribution shown. Check the
  provider's terms before publishing results. Do not use Google Maps/Earth tiles here.
  To use another provider, set environment variables before starting the app:
  `TILE_URL` (with {z}/{x}/{y}), `TILE_ATTRIBUTION`, `MAX_ZOOM` (default 18).
  Some rural areas have no imagery at the highest zooms; if tiles look blank, lower `MAX_ZOOM`.
- Own GPU PC instead of Colab: install torch + segment-anything, put the SAM checkpoint next
  to `detector/detector_service.py`, run `uvicorn detector_service:app --port 8001` inside
  `detector/`, and paste `http://127.0.0.1:8001` as the connection link.
- If Colab disconnects, run the last cell again and paste the new link.
- Anyone who can open the web app can use your detector link. Run it on your own computer for
  now; add logins before putting it on the internet.

## Files
- `app/main.py` API, database, area jobs, export
- `app/tiles.py` tile download, stitching, georeferencing
- `app/learner.py` features, training, guessing, honest accuracy tests
- `detector/detector_service.py` the SAM pipeline as a job server
- `colab/parcel_detector_colab.ipynb` the same thing, ready for Colab
  (rebuild with `python colab/build_notebook.py` after editing the detector)
- `static/index.html` the web page (Leaflet is bundled in `static/vendor`)
- `tests/mock_tiles.py` stand-in tile server used for testing (not real imagery)
