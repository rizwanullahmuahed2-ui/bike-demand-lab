# Data provenance

Source: [UCI Bike Sharing, dataset 275](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). Creator: Hadi Fanaee-T. Dataset DOI: **10.24432/C5W894**. UCI lists the dataset under **CC BY 4.0**; this also applies to the derived held-out prediction rows and aggregated measurements distributed here.

The observations describe Capital Bikeshare rentals in Washington, D.C. during 2011–2012. The hourly table contains 17,379 records. The source includes observed weather, normalized temperature/humidity/wind fields, calendar fields and counts. Weather is not a forecast supplied before the predicted hour. Source time labels have no UTC offset; this project preserves them without inventing a time zone.

Archive: `https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip`

- Archive SHA-256: `b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401`
- `hour.csv` SHA-256: `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`

The acquisition command verifies both hashes. Raw downloads are excluded from Git. This implementation derives timestamps, applies an explicit feature allowlist, splits dates and calculates model predictions; it does not alter the original target counts.

Related publication: Fanaee-T, H. and Gama, J., *Event labeling combining ensemble detectors and background knowledge*, DOI **10.1007/s13748-013-0040-3**. Cite the source dataset and original work if reusing these observations.
