# data/

Git-ignored except this file.

| Path | Contents | Created by |
|---|---|---|
| `data/downloads/` | COCO zips (val2017 images ≈780 MB, annotations ≈240 MB), kept on Drive so they download only once | NB01 |
| `data/custom_images/` | Optional: your own CCTV / traffic frames for the qualitative run | you (NB11) |

On Colab the extracted images live on the fast local disk `/content/data/coco/` (not Drive) and are re-extracted from the zips after each runtime restart. Running locally, they go to `data/coco/`.
