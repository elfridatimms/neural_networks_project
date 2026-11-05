# Tim 07 – Segmentacija krvnih žila mrežnice (Neuronske mreže, FER 2026./2027.)

Kod za 2. termin: priprema podataka, podjela train/val/test i analiza skupova FIVES i CHASE_DB1.

## Struktura

```
projekt/
├── data/                         # podaci (NISU u arhivi s kodom, vidi dolje)
│   ├── FIVES/{train,test}/{Original,Ground truth}/
│   ├── FIVES/Quality Assessment.xlsx
│   └── CHASE_DB1_raw/data/*.parquet
└── retinal-vessel-segmentation/
    ├── src/
    │   ├── config.py             # putanje, seed, udjeli val/test
    │   ├── prepare_data.py       # popis slika, izbacivanje praznih maski, podjela 70/15/15
    │   └── data_analysis.py      # statistike po slici + svi grafovi iz izvještaja
    ├── notebooks/
    │   └── 01_analiza_podataka.ipynb   # analiza korak po korak (s izlazima)
    ├── splits/                   # index.csv, split.csv (fiksna podjela, seed 42)
    └── results/
        ├── figures/              # grafovi korišteni u izvještaju
        └── metrics/              # image_stats.csv, summary_stats.csv, near_duplicates.csv
```

## Pokretanje

1. Preuzeti podatke u `projekt/data/`:
   - FIVES: https://figshare.com/articles/dataset/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169
     (raspakirati `.rar` i preimenovati mapu u `FIVES`)
   - CHASE_DB1:
     ```bash
     python -c "from huggingface_hub import snapshot_download; snapshot_download('MedOtter/CHASE_DB1', repo_type='dataset', local_dir='../data/CHASE_DB1_raw')"
     ```
2. Instalirati pakete: `pip install -r requirements.txt`
3. Iz mape `retinal-vessel-segmentation/`:
   ```bash
   python src/prepare_data.py     # -> splits/split.csv, data/CHASE_DB1/
   python src/data_analysis.py    # -> results/ (~4 min)
   ```
   ili otvoriti `notebooks/01_analiza_podataka.ipynb`.

Podjela je deterministička (seed 42), pa svi članovi tima dobivaju istu `split.csv`.
