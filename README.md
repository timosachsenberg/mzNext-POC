Currently just a playground to quickly test a few ideas regarding a potential parquet format for MS data.
test files are available at the release

Observations:
1. long format is compressed a bit larger than the conpressed mzML. Random access/chrom. extraction performance is super slow (up to 200s for extraction)
2. try one row per spectrum / with data array (e.g. m/z) in cell
```
Load experiment from mzML: 6.477532863616943 seconds
Create json representaiton of meta data: 3.0441911220550537 seconds
Writing parquet files: 25.218493938446045 seconds
Accessing 100 random spectra: 0.002962350845336914 seconds
Extracting 100 random m/z ranges (2 min, 1 m/z) from MS1 spectra: 0.216780424118042 seconds
Extracting 100 random m/z ranges (2 min, 1 m/z) from MS2 spectra: 0.0374913215637207 seconds
```
