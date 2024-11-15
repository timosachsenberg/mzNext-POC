Currently just a playground to quickly test a few ideas regarding a potential parquet format for MS data.
test files are available at the release

Observations:
1. long format is compressed a bit larger than the conpressed mzML. Random access/chrom. extraction performance is super slow (up to 200s for extraction)
2. try one row per spectrum / with data array (e.g. m/z) in cell
```
Load experiment from mzML: 6.359102725982666 seconds
Create json representaiton of meta data: 2.91546368598938 seconds
Writing parquet files: 24.772038221359253 seconds
Loading spectra parquet files: 3.0158519744873047 seconds
Loading chromatogram parquet files: 2.946080446243286 seconds
Accessing 100 random spectra: 0.004001140594482422 seconds
MS1: Extracted a total of 1250295 peaks from the m/z and rt ranges.
MS1: Total time for extracting peaks from m/z and rt ranges: 3.1221811771392822 seconds
MS2: Extracted a total of 1417186 peaks from the m/z and rt ranges.
MS2: Total time for extracting peaks from m/z and rt ranges: 4.18139123916626 seconds
```
