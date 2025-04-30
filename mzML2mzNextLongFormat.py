import json
import argparse
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyopenms import MSExperiment, MzMLFile, IonSource, MassAnalyzer, ChecksumType

#TODO: use cv terms where appropriate
#TODO: add example for each vendor
#TODO: add example with ion mobility
#TODO: think if having other data types make sense (e.g., mass trace, mobilogram, IM frames)
#TODO: add imaging MS example (e.g., additional collumns)
#TODO: evaluate what meta data should go into json or in columns

def fillMetaData(exp):
    start_time = time.time()

    metadata = {}

    # Instrument metadata
    instrument = exp.getInstrument()
    instrument_meta = {
        'name': instrument.getName(),
        'vendor': instrument.getVendor(),
        'model': instrument.getModel(),
        'customizations': instrument.getCustomizations(),
        'ion_optics': str(instrument.getIonOptics()),
        'software': {
            'name': instrument.getSoftware().getName(),
            'version': instrument.getSoftware().getVersion(),
        },
    }

    # Ion sources
    ion_sources = []

    # Enum mappings for IonSource
    polarity_mapping = {
        IonSource.Polarity.POLNULL: 'UNKNOWN',
        IonSource.Polarity.POSITIVE: 'POSITIVE',
        IonSource.Polarity.NEGATIVE: 'NEGATIVE',
    }

    inlet_type_mapping = {
        IonSource.InletType.INLETNULL: 'UNKNOWN',
        IonSource.InletType.DIRECT: 'DIRECT',
        IonSource.InletType.BATCH: 'BATCH',
        IonSource.InletType.CHROMATOGRAPHY: 'CHROMATOGRAPHY',
        IonSource.InletType.PARTICLEBEAM: 'PARTICLEBEAM',
        IonSource.InletType.MEMBRANESEPARATOR: 'MEMBRANESEPARATOR',
        IonSource.InletType.OPENSPLIT: 'OPENSPLIT',
        IonSource.InletType.JETSEPARATOR: 'JETSEPARATOR',
        IonSource.InletType.SEPTUM: 'SEPTUM',
        IonSource.InletType.RESERVOIR: 'RESERVOIR',
        IonSource.InletType.MOVINGBELT: 'MOVINGBELT',
        IonSource.InletType.MOVINGWIRE: 'MOVINGWIRE',
        IonSource.InletType.FLOWINJECTIONANALYSIS: 'FLOWINJECTIONANALYSIS',
        IonSource.InletType.ELECTROSPRAYINLET: 'ELECTROSPRAYINLET',
        IonSource.InletType.THERMOSPRAYINLET: 'THERMOSPRAYINLET',
        IonSource.InletType.INFUSION: 'INFUSION',
        IonSource.InletType.CONTINUOUSFLOWFASTATOMBOMBARDMENT: 'CONTINUOUSFLOWFASTATOMBOMBARDMENT',
        IonSource.InletType.INDUCTIVELYCOUPLEDPLASMA: 'INDUCTIVELYCOUPLEDPLASMA',
        IonSource.InletType.MEMBRANE: 'MEMBRANE',
        IonSource.InletType.NANOSPRAY: 'NANOSPRAY',
    }

    ionization_method_mapping = {
        IonSource.IonizationMethod.IONMETHODNULL: 'UNKNOWN',
        IonSource.IonizationMethod.ESI: 'ESI',
        IonSource.IonizationMethod.EI: 'EI',
        IonSource.IonizationMethod.CI: 'CI',
        IonSource.IonizationMethod.FAB: 'FAB',
        IonSource.IonizationMethod.TSP: 'TSP',
        IonSource.IonizationMethod.LD: 'LD',
        IonSource.IonizationMethod.FD: 'FD',
        IonSource.IonizationMethod.FI: 'FI',
        IonSource.IonizationMethod.PD: 'PD',
        IonSource.IonizationMethod.SI: 'SI',
        IonSource.IonizationMethod.TI: 'TI',
        IonSource.IonizationMethod.API: 'API',
        IonSource.IonizationMethod.ISI: 'ISI',
        IonSource.IonizationMethod.CID: 'CID',
        IonSource.IonizationMethod.CAD: 'CAD',
        IonSource.IonizationMethod.HN: 'HN',
        IonSource.IonizationMethod.APCI: 'APCI',
        IonSource.IonizationMethod.APPI: 'APPI',
        IonSource.IonizationMethod.ICP: 'ICP',
        IonSource.IonizationMethod.NESI: 'NESI',
        IonSource.IonizationMethod.MESI: 'MESI',
        IonSource.IonizationMethod.SELDI: 'SELDI',
        IonSource.IonizationMethod.SEND: 'SEND',
        IonSource.IonizationMethod.FIB: 'FIB',
        IonSource.IonizationMethod.MALDI: 'MALDI',
        IonSource.IonizationMethod.MPI: 'MPI',
        IonSource.IonizationMethod.DI: 'DI',
        IonSource.IonizationMethod.FA: 'FA',
        IonSource.IonizationMethod.FII: 'FII',
        IonSource.IonizationMethod.GD_MS: 'GD_MS',
        IonSource.IonizationMethod.NICI: 'NICI',
        IonSource.IonizationMethod.NRMS: 'NRMS',
        IonSource.IonizationMethod.PI: 'PI',
        IonSource.IonizationMethod.PYMS: 'PYMS',
        IonSource.IonizationMethod.REMPI: 'REMPI',
        IonSource.IonizationMethod.AI: 'AI',
        IonSource.IonizationMethod.ASI: 'ASI',
        IonSource.IonizationMethod.AD: 'AD',
        IonSource.IonizationMethod.AUI: 'AUI',
        IonSource.IonizationMethod.CEI: 'CEI',
        IonSource.IonizationMethod.CHEMI: 'CHEMI',
        IonSource.IonizationMethod.DISSI: 'DISSI',
        IonSource.IonizationMethod.LSI: 'LSI',
        IonSource.IonizationMethod.PEI: 'PEI',
        IonSource.IonizationMethod.SOI: 'SOI',
        IonSource.IonizationMethod.SPI: 'SPI',
        IonSource.IonizationMethod.SUI: 'SUI',
        IonSource.IonizationMethod.VI: 'VI',
        IonSource.IonizationMethod.AP_MALDI: 'AP_MALDI',
        IonSource.IonizationMethod.SILI: 'SILI',
        IonSource.IonizationMethod.SALDI: 'SALDI',
    }

    for source in instrument.getIonSources():
        # Retrieve enum values
        polarity = source.getPolarity()
        inlet_type = source.getInletType()
        ionization_method = source.getIonizationMethod()

        # Map enum values to strings
        polarity_str = polarity_mapping.get(polarity, 'UNKNOWN')
        inlet_type_str = inlet_type_mapping.get(inlet_type, 'UNKNOWN')
        ionization_method_str = ionization_method_mapping.get(ionization_method, 'UNKNOWN')

        source_meta = {
            'order': source.getOrder(),
            'polarity': polarity_str,
            'ionization_method': ionization_method_str,
            'inlet_type': inlet_type_str,
        }
        ion_sources.append(source_meta)

    instrument_meta['ion_sources'] = ion_sources

    # Mass analyzers
    # Mass analyzers
    mass_analyzers = []
    for analyzer in instrument.getMassAnalyzers():
        # Map enum values to their string representations
        analyzer_type_mapping = {
            MassAnalyzer.AnalyzerType.ANALYZERNULL: 'UNKNOWN',
            MassAnalyzer.AnalyzerType.QUADRUPOLE: 'QUADRUPOLE',
            MassAnalyzer.AnalyzerType.PAULIONTRAP: 'PAULIONTRAP',
            MassAnalyzer.AnalyzerType.RADIALEJECTIONLINEARIONTRAP: 'RADIALEJECTIONLINEARIONTRAP',
            MassAnalyzer.AnalyzerType.AXIALEJECTIONLINEARIONTRAP: 'AXIALEJECTIONLINEARIONTRAP',
            MassAnalyzer.AnalyzerType.TOF: 'TOF',
            MassAnalyzer.AnalyzerType.SECTOR: 'SECTOR',
            MassAnalyzer.AnalyzerType.FOURIERTRANSFORM: 'FOURIERTRANSFORM',
            MassAnalyzer.AnalyzerType.IONSTORAGE: 'IONSTORAGE',
            MassAnalyzer.AnalyzerType.ESA: 'ESA',
            MassAnalyzer.AnalyzerType.IT: 'IT',
            MassAnalyzer.AnalyzerType.SWIFT: 'SWIFT',
            MassAnalyzer.AnalyzerType.CYCLOTRON: 'CYCLOTRON',
            MassAnalyzer.AnalyzerType.ORBITRAP: 'ORBITRAP',
            MassAnalyzer.AnalyzerType.LIT: 'LIT',
        }

        resolution_method_mapping = {
            MassAnalyzer.ResolutionMethod.RESMETHNULL: 'UNKNOWN',
            MassAnalyzer.ResolutionMethod.FWHM: 'FWHM',
            MassAnalyzer.ResolutionMethod.TENPERCENTVALLEY: 'TENPERCENTVALLEY',
            MassAnalyzer.ResolutionMethod.BASELINE: 'BASELINE',
        }

        resolution_type_mapping = {
            MassAnalyzer.ResolutionType.RESTYPENULL: 'UNKNOWN',
            MassAnalyzer.ResolutionType.CONSTANT: 'CONSTANT',
            MassAnalyzer.ResolutionType.PROPORTIONAL: 'PROPORTIONAL',
        }

        scan_direction_mapping = {
            MassAnalyzer.ScanDirection.SCANDIRNULL: 'UNKNOWN',
            MassAnalyzer.ScanDirection.UP: 'UP',
            MassAnalyzer.ScanDirection.DOWN: 'DOWN',
        }

        scan_law_mapping = {
            MassAnalyzer.ScanLaw.SCANLAWNULL: 'UNKNOWN',
            MassAnalyzer.ScanLaw.EXPONENTIAL: 'EXPONENTIAL',
            MassAnalyzer.ScanLaw.LINEAR: 'LINEAR',
            MassAnalyzer.ScanLaw.QUADRATIC: 'QUADRATIC',
        }

        reflectron_state_mapping = {
            MassAnalyzer.ReflectronState.REFLSTATENULL: 'UNKNOWN',
            MassAnalyzer.ReflectronState.ON: 'ON',
            MassAnalyzer.ReflectronState.OFF: 'OFF',
            MassAnalyzer.ReflectronState.NONE: 'NONE',
        }

        # Retrieve enum values
        analyzer_type = analyzer.getType()
        resolution_method = analyzer.getResolutionMethod()
        resolution_type = analyzer.getResolutionType()
        scan_direction = analyzer.getScanDirection()
        scan_law = analyzer.getScanLaw()
        reflectron_state = analyzer.getReflectronState()

        # Create the mass analyzer metadata dictionary
        analyzer_meta = {
            'order': analyzer.getOrder(),
            'type': analyzer_type_mapping.get(analyzer_type, 'UNKNOWN'),
            'resolution_method': resolution_method_mapping.get(resolution_method, 'UNKNOWN'),
            'resolution_type': resolution_type_mapping.get(resolution_type, 'UNKNOWN'),
            'scan_direction': scan_direction_mapping.get(scan_direction, 'UNKNOWN'),
            'scan_law': scan_law_mapping.get(scan_law, 'UNKNOWN'),
            'reflectron_state': reflectron_state_mapping.get(reflectron_state, 'UNKNOWN'),
            'resolution': analyzer.getResolution(),
            'accuracy': analyzer.getAccuracy(),
            'scan_rate': analyzer.getScanRate(),
            'scan_time': analyzer.getScanTime(),
            'tof_total_path_length': analyzer.getTOFTotalPathLength(),
            'isolation_width': analyzer.getIsolationWidth(),
            'final_MS_exponent': analyzer.getFinalMSExponent(),
            'magnetic_field_strength': analyzer.getMagneticFieldStrength(),
        }
        mass_analyzers.append(analyzer_meta)

    instrument_meta['mass_analyzers'] = mass_analyzers


    # Ion detectors
    ion_detectors = []
    for detector in instrument.getIonDetectors():
        detector_meta = {
            'order': detector.getOrder(),
            'type': str(detector.getType()),
            'acquisition_mode': str(detector.getAcquisitionMode()),
            'resolution': detector.getResolution(),
            'adc_sampling_frequency': detector.getADCSamplingFrequency(),
        }
        ion_detectors.append(detector_meta)
    instrument_meta['ion_detectors'] = ion_detectors

    metadata['instrument'] = instrument_meta

    # Source files
    source_files = []

    # Enum mapping for ChecksumType
    checksum_type_mapping = {
        ChecksumType.UNKNOWN_CHECKSUM: 'UNKNOWN_CHECKSUM',
        ChecksumType.SHA1: 'SHA1',
        ChecksumType.MD5: 'MD5',
    }

    for source_file in exp.getSourceFiles():
        checksum_type = source_file.getChecksumType()
        checksum_type_str = checksum_type_mapping.get(checksum_type, 'UNKNOWN_CHECKSUM')
        source_file_meta = {
            'name_of_file': source_file.getNameOfFile(),
            'path_to_file': source_file.getPathToFile(),
            'file_size': source_file.getFileSize(),
            'file_type': source_file.getFileType(),
            'checksum': source_file.getChecksum(),
            'checksum_type': checksum_type_str,
            'native_id_type': source_file.getNativeIDType(),
            'native_id_type_accession': source_file.getNativeIDTypeAccession(),
        }
        source_files.append(source_file_meta)

    metadata['source_files'] = source_files

    # Experiment-level metadata
    exp_keys = []
    exp.getKeys(exp_keys)
    exp_meta = {key.decode('utf-8'): exp.getMetaValue(key) for key in exp_keys}
    metadata['experiment_meta'] = exp_meta

    # Spectra metadata
    spectra_meta = []
    for spectrum_index, spectrum in enumerate(exp.getSpectra()):
        spectrum_keys = []
        spectrum.getKeys(spectrum_keys)
        spectrum_meta_values = {key.decode('utf-8'): spectrum.getMetaValue(key) for key in spectrum_keys}

        # Precursors
        precursors = []
        for precursor in spectrum.getPrecursors(): # should be zero or one
            precursor_keys = []
            precursor.getKeys(precursor_keys)
            precursor_meta_values = {key.decode('utf-8'): precursor.getMetaValue(key) for key in precursor_keys}
            precursor_index = exp.getPrecursorSpectrum(spectrum_index)
            precursor_meta = {
                'rt': exp[precursor_index].getRT(),
                'mz': precursor.getMZ(),
                'charge': precursor.getCharge(),
                'intensity': precursor.getIntensity(),
                'selected_ion_mz': precursor.getMZ(),
                'selected_ion_charge': precursor.getCharge() if precursor.getCharge() != 0 else None,
                'selected_ion_intensity': precursor.getIntensity() if precursor.getIntensity() != 0 else None,
                'isolation_window_target': precursor.getMZ(),
                'isolation_window_lower_offset': precursor.getIsolationWindowLowerOffset(),
                'isolation_window_upper_offset': precursor.getIsolationWindowUpperOffset(),
                'activation_methods': [str(method) for method in precursor.getActivationMethods()],
                'activation_energy': precursor.getActivationEnergy(),
                'meta_values': precursor_meta_values,
            }
            precursors.append(precursor_meta)

        # Products
        products = []
        for product in spectrum.getProducts():
            product_meta = {
                'mz': product.getMZ(),
                'isolation_window_lower_offset': product.getIsolationWindowLowerOffset(),
                'isolation_window_upper_offset': product.getIsolationWindowUpperOffset(),
            }
            products.append(product_meta)

        spectrum_meta = {
            'native_id': spectrum.getNativeID(),
            'rt': spectrum.getRT(),
            'ms_level': spectrum.getMSLevel(),
            'meta_values': spectrum_meta_values,
            'precursors': precursors,
            'products': products,
        }
        spectra_meta.append(spectrum_meta)
    metadata['spectra'] = spectra_meta

    # Chromatograms metadata
    chromatograms_meta = []
    for chromatogram in exp.getChromatograms():
        chromatogram_keys = []
        chromatogram.getKeys(chromatogram_keys)
        chromatogram_meta_values = {key.decode('utf-8'): chromatogram.getMetaValue(key) for key in chromatogram_keys}
        chromatogram_meta = {
            'native_id': chromatogram.getNativeID(),
            'meta_values': chromatogram_meta_values,
            'precursor': {'mz': chromatogram.getPrecursor().getMZ()},
            'product': {'mz': chromatogram.getProduct().getMZ()},
        }
        chromatograms_meta.append(chromatogram_meta)
    metadata['chromatograms'] = chromatograms_meta

    # Convert metadata to JSON
    json_str = json.dumps(metadata, indent=2)

    end_time = time.time()
    print(f"Create json representaiton of meta data: {end_time - start_time} seconds")

    return json_str



def write(json_str):
    # Create a list to hold spectrum data for the DataFrame
    spectra_data = []

    # Iterate over all spectra
    for spectrum in exp.getSpectra():
        # Extract spectrum-level metadata
        spectrum_id = spectrum.getNativeID()
        ms_level = spectrum.getMSLevel()
        rt = spectrum.getRT()
        
        ion_mobility = None
        if spectrum.getDriftTime() != 0:
            ion_mobility = spectrum.getDriftTime()
        
        # mz and intensity arrays
        mz_array = spectrum.get_peaks()[0]
        intensity_array = spectrum.get_peaks()[1]
        
        precursors = spectrum.getPrecursors()

        # Iterate over peaks to create long-format data
        for mz, intensity in zip(mz_array, intensity_array):
            spectra_data.append({
                'id': spectrum_id,
                'data_type': 'spectrum',  # New field to distinguish data type
                'ms_level': ms_level,
                'rt': rt, # scan start time
                'mz': mz, 
                'intensity': intensity,
                'ion_mobility': ion_mobility,    
            })

    # Process chromatograms
    for chromatogram in exp.getChromatograms():
        chrom_id = chromatogram.getNativeID()
        time_array = chromatogram.get_peaks()[0]
        intensity_array = chromatogram.get_peaks()[1]
        
        # Get precursor and product information
        precursor = chromatogram.getPrecursor()
        product = chromatogram.getProduct()
        
        for rt, intensity in zip(time_array, intensity_array):
            spectra_data.append({
                'id': chrom_id,
                'data_type': 'chromatogram',  # Distinguish chromatogram data
                'ms_level': None,  # Chromatograms don't have MS level
                'rt': rt,
                'mz': product.getMZ(),
                'intensity': intensity,
                'ion_mobility': None,
            })

    # Observation: having variable mz's could model a data_type 'mass trace'
    # Convert to DataFrame
    combined_df = pd.DataFrame(spectra_data)

    # Optimize data types
    combined_df['id'] = combined_df['id'].astype('category')
    combined_df['data_type'] = combined_df['data_type'].astype('category')
    combined_df['ms_level'] = combined_df['ms_level'].astype('Int8')  # nullable integer
#    combined_df['ion_injection_time'] = combined_df['ion_injection_time'].astype('float32')
#    combined_df['selected_ion_mz'] = combined_df['selected_ion_mz'].astype('float32')
#    combined_df['selected_ion_charge'] = combined_df['selected_ion_charge'].astype('float32')
#    combined_df['selected_ion_intensity'] = combined_df['selected_ion_intensity'].astype('float32')
    combined_df['rt'] = combined_df['rt'].astype('float32')
    combined_df['mz'] = combined_df['mz'].astype('float32')
    combined_df['intensity'] = combined_df['intensity'].astype('float32')
    combined_df['ion_mobility'] = combined_df['ion_mobility'].astype('float32')
 #   combined_df['spectrum_ref'] = combined_df['spectrum_ref'].astype('category')

    # Write combined data to single Parquet file
    combined_table = pa.Table.from_pandas(combined_df)
    existing_meta = combined_table.schema.metadata

    # Corrected metadata structure
    combined_meta = {
        b"mzML_metadata": json_str.encode(),  # Specify key-value structure correctly
        **(existing_meta or {})  # Merge with existing metadata, if any
    }

    # Rebuild the table with the new schema that includes updated metadata
    combined_table = combined_table.replace_schema_metadata(combined_meta)

    start_time = time.time()
    pq.write_table(combined_table, f"{args.output_file}.parquet",
        compression='snappy',
        use_dictionary=True,
        write_statistics=True,
        )
    end_time = time.time()
    print(f"Writing parquet file: {end_time - start_time} seconds")

    start_time = time.time()
    combined_table = pd.read_parquet(f"{args.output_file}.parquet", engine='pyarrow')
    end_time = time.time()
    print(f"Loading parquet file into memory: {end_time - start_time} seconds")

def benchmark():
    import numpy as np
    import polars as pl
    import time

    # Create lazy DataFrame from Parquet file
    start_time = time.time()
    df_lazy = pl.scan_parquet(f"{args.output_file}.parquet")
    end_time = time.time()
    print(f"Creating lazy dataframe (no data loaded into memory): {end_time - start_time} seconds")

    # Task 1: Access 100 random spectra

    # Collect unique IDs
    spectra_ids = df_lazy.filter(pl.col('data_type') == 'spectrum').select('id').unique().collect()['id'].to_list()
    num_spectra = len(spectra_ids)

    sum_time = 0.0
    if num_spectra >= 100:
        random_ids = np.random.choice(spectra_ids, 100, replace=False)
        t0 = time.time()
        result = (df_lazy
            .filter(pl.col('data_type') == 'spectrum')
            .filter(pl.col('id').is_in(random_ids))
            .collect())
        t1 = time.time()
        sum_time += t1 - t0        
    else:
        print("Not enough spectra to perform the benchmark.")
    
    print(f"Average time to access a random spectrum by native id: {sum_time / 100.0} seconds")

    # Benchmark common data access pattern: extract 100 random m/z ranges from MS1 and MS2 spectra

    # Task 2: Extract random m/z ranges of size 0.2 m/z and +-60 seconds RT from MS1 and MS2 spectra
    for ms_level in [1, 2]:
        sum_time = 0
        for _ in range(100):
            mz = np.random.uniform(200, 2000)
            mz_tol = 0.1 # 0.1 m/z tolerance
            rt = np.random.uniform(100, 1100)
            rt_tol = 60.0

            t0 = time.time()
            df = df_lazy.filter(
                (pl.col("rt").is_between(rt - rt_tol,  rt + rt_tol))
                & (pl.col("mz").is_between(mz - mz_tol, mz + mz_tol))
                & (pl.col("data_type") == "spectrum")
                & (pl.col("ms_level") == ms_level)
                ).collect()
            t1 = time.time()
            sum_time += t1 - t0
        print(f"Average time to extract a random m/z (+-{mz_tol}),rt (+-{rt_tol}) range from MS{ms_level} spectra: {sum_time / 100.0} seconds")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert mzML metadata to JSON')
    parser.add_argument('input_file', help='Input mzML file')
    parser.add_argument('-o', '--output_file', default='metadata.json', help='Output JSON file')
    args = parser.parse_args()

    # Load the mzML file
    start_time = time.time()
    exp = MSExperiment()
    MzMLFile().load(args.input_file, exp)  # Use filename from command line argument
    end_time = time.time()
    print(f"Load experiment from mzML: {end_time - start_time} seconds")

    json_str = fillMetaData(exp)
    write(json_str)
    benchmark()
