Currently just a playground to quickly test a few ideas regarding a potential parquet format for MS data.
test files are available at the release

Long format:
```
Load experiment from mzML: 5.796112775802612 seconds
Create json representation of meta data: 3.2389822006225586 seconds
Writing parquet file: 24.799195051193237 seconds
Loading parquet file into memory: 7.5644850730896 seconds
Creating lazy dataframe (no data loaded into memory): 0.0009696483612060547 seconds
Average time to access a random spectrum: 0.0037169885635375977 seconds
Average time to extract a random m/z (+-0.1),rt (+-60.0) range from MS1 spectra: 0.058548598289489745 seconds
Average time to extract a random m/z (+-0.1),rt (+-60.0) range from MS2 spectra: 0.05765039205551147 seconds
```
