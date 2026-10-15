# Geoportal building-dataset scripts

These scripts download images of buildings from the Geoportal. They also download the corresponding building outlines and create masks.

## Usage

CIR:

```bash
python build_dataset_cir.py --lat1 51.10 --lon1 17.00 --lat2 51.11 --lon2 17.02 --output cir_dataset
```

High resolution:

```bash
python build_dataset_high_res.py --lat1 51.10 --lon1 17.00 --lat2 51.11 --lon2 17.02 --output high_res_dataset
```

Replace the example coordinates with the two WGS84 corners for your area. Run either script with `--help` to see all supported options.


### Warning

The downloaded datasets might differ, beacause of avaiblity of either cir or high res images.