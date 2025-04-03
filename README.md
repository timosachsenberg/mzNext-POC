Currently just a playground to quickly test a few ideas regarding a potential parquet format for MS data.
test files are available at the release

Long format:
```
python mzML2mzNextLongFormat.py /mnt/d/Tutorials/Example_Data/ProteomicsLFQ/UPS1_50000amol_R1.mzML 
Load experiment from mzML: 38.55868220329285 seconds
Create json representaiton of meta data: 2.45767879486084 seconds
Writing parquet file: 4.293434143066406 seconds
Loading parquet file into memory: 3.498243570327759 seconds
Creating lazy dataframe (no data loaded into memory): 0.003572702407836914 seconds
Average time to access a random spectrum by native id: 0.0038401222229003905 seconds
Average time to extract a random m/z (+-0.1),rt (+-60.0) range from MS1 spectra: 0.38623351335525513 seconds
Average time to extract a random m/z (+-0.1),rt (+-60.0) range from MS2 spectra: 0.3775592088699341 seconds

609628424 Mar 19 17:40 metadata.json.parquet
767043386 Aug 20  2021 /mnt/d/Tutorials/Example_Data/ProteomicsLFQ/UPS1_50000amol_R1.mzML
```
