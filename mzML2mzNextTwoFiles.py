import json
import argparse
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pyopenms import *
import numpy as np

#TODO: use cv terms where appropriate
#TODO: add example for each vendor
#TODO: add example with ion mobility
#TODO: think if having other data types make sense (e.g., mass trace, mobilogram, IM frames)
#TODO: add imaging MS example (e.g., additional collumns)
#TODO: evaluate what meta data should go into json or in columns

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


def fillMetaDate(exp):
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
    print(f"Create json representation of meta data: {end_time - start_time} seconds")

    return json_str


def writeParquet(json_str):
    # Create lists to hold spectrum and chromatogram data separately
    spectra_data = []
    chromatogram_data = []

    # Process spectra
    for spectrum in exp.getSpectra():
        spectrum_id = spectrum.getNativeID()
        ms_level = spectrum.getMSLevel()
        rt = spectrum.getRT()
        
        ion_mobility = None
        if spectrum.getDriftTime() != 0:
            ion_mobility = spectrum.getDriftTime()
        
        # Get peaks as numpy arrays
        mz_array, intensity_array = spectrum.get_peaks()
        
        # Store complete spectrum data
        spectra_data.append({
            'id': spectrum_id,
            'data_type': 'spectrum',
            'ms_level': ms_level,
            'rt': rt,
            'ion_mobility': ion_mobility,
            'mz_array': mz_array,
            'intensity_array': intensity_array
        })

    # Process chromatograms
    for chromatogram in exp.getChromatograms():
        chrom_id = chromatogram.getNativeID()
        time_array, intensity_array = chromatogram.get_peaks()
        
        # Get precursor and product information
        precursor = chromatogram.getPrecursor()
        product = chromatogram.getProduct()
        
        chromatogram_data.append({
            'id': chrom_id,
            'data_type': 'chromatogram',
            'precursor_mz': precursor.getMZ(),
            'product_mz': product.getMZ(),
            'time_array': time_array,
            'intensity_array': intensity_array
        })

    # Convert to DataFrames with array columns
    spectra_df = pd.DataFrame(spectra_data)
    chromatogram_df = pd.DataFrame(chromatogram_data)

    # Optimize data types for non-array columns
    spectra_df['id'] = spectra_df['id'].astype('category')
    spectra_df['data_type'] = spectra_df['data_type'].astype('category')
    spectra_df['ms_level'] = spectra_df['ms_level'].astype('Int8')
    spectra_df['rt'] = spectra_df['rt'].astype('float32')
    spectra_df['ion_mobility'] = spectra_df['ion_mobility'].astype('float32')

    chromatogram_df['id'] = chromatogram_df['id'].astype('category')
    chromatogram_df['data_type'] = chromatogram_df['data_type'].astype('category')
    chromatogram_df['precursor_mz'] = chromatogram_df['precursor_mz'].astype('float32')
    chromatogram_df['product_mz'] = chromatogram_df['product_mz'].astype('float32')

    # Convert to PyArrow tables
    spectra_table = pa.Table.from_pandas(spectra_df)
    chromatogram_table = pa.Table.from_pandas(chromatogram_df)

    # Add metadata
    combined_meta = {
        b"mzML_metadata": json_str.encode()
    }

    # Update schema metadata
    spectra_table = spectra_table.replace_schema_metadata(combined_meta)
    chromatogram_table = chromatogram_table.replace_schema_metadata(combined_meta)

    # Write tables to Parquet files
    start_time = time.time()
    
    pq.write_table(spectra_table, 
                   f"{args.output_file}_spectra.parquet",
                   compression='snappy',
                   use_dictionary=True,
                   write_statistics=True)
    
    pq.write_table(chromatogram_table, 
                   f"{args.output_file}_chromatograms.parquet",
                   compression='snappy',
                   use_dictionary=True,
                   write_statistics=True)
                   
    end_time = time.time()
    print(f"Writing parquet files: {end_time - start_time} seconds")

def benchmarkParquet():
    import numpy as np
    import pyarrow.dataset as ds

    # Load datasets
    start_time = time.time()
    spectra_dataset = ds.dataset(f"{args.output_file}_spectra.parquet", format="parquet")
    end_time = time.time()
    print(f"Loading spectra parquet files: {end_time - start_time} seconds")

    start_time = time.time()
    chromatogram_dataset = ds.dataset(f"{args.output_file}_chromatograms.parquet", format="parquet")
    end_time = time.time()
    print(f"Loading chromatogram parquet files: {end_time - start_time} seconds")

    # Function to collect unique IDs
    def get_unique_ids(filtered_dataset):
        unique_ids = set()
        for batch in filtered_dataset.to_batches(columns=['id']):
            ids_in_batch = batch.column('id').to_pylist()
            unique_ids.update(ids_in_batch)
        return list(unique_ids)

    # Get spectrum counts
    all_spectra = spectra_dataset.to_table()
    ms1_spectra = spectra_dataset.filter(ds.field('ms_level') == 1).to_table()
    ms2_spectra = spectra_dataset.filter(ds.field('ms_level') == 2).to_table()

    num_spectra = len(all_spectra)
    num_ms1_spectra = len(ms1_spectra)
    num_ms2_spectra = len(ms2_spectra)

    # First Task: Access 100 random spectra
    start_time = time.time()
    if num_spectra >= 100:
        random_indices = np.random.choice(num_spectra, 100, replace=False)
        selected_spectra = all_spectra.take(random_indices)
    else:
        print("Not enough spectra to perform the benchmark.")
    end_time = time.time()
    print(f"Accessing 100 random spectra: {end_time - start_time} seconds")


    # Filter spectra by ms_level
    ms1_spectra_table = all_spectra.filter(pc.equal(all_spectra['ms_level'], 1))
    ms2_spectra_table = all_spectra.filter(pc.equal(all_spectra['ms_level'], 2))

    # Get the retention time ranges for ms_level 1 and 2
    def get_rt_range(spectra_table):
        rt_array = spectra_table['rt'].to_pandas().values
        return rt_array.min(), rt_array.max()

    ms1_min_rt, ms1_max_rt = get_rt_range(ms1_spectra_table) if len(ms1_spectra_table) > 0 else (0, 0)
    ms2_min_rt, ms2_max_rt = get_rt_range(ms2_spectra_table) if len(ms2_spectra_table) > 0 else (0, 0)

    # Define m/z and rt range sizes
    mz_range_size = 50.0  # Size of the m/z range
    rt_range_size = 120.0  # Size of the rt range

    # Generate 100 m/z ranges between 400 and 1400
    mz_start_values = np.random.uniform(400, 1400 - mz_range_size, 100)
    mz_ranges = [(start, start + mz_range_size) for start in mz_start_values]

    # Function to perform the extraction and benchmarking
    def extract_peaks(spectra_table, min_rt, max_rt, ms_level_label):
        if len(spectra_table) == 0:
            print(f"No MS{ms_level_label} spectra available for benchmarking.")
            return

        # Generate 100 rt ranges randomly within the available rt range
        rt_start_values = np.random.uniform(min_rt, max_rt - rt_range_size, 100)
        rt_ranges = [(start, start + rt_range_size) for start in rt_start_values]

        # Start benchmarking
        start_time = time.time()

        total_peaks_extracted = 0  # To keep track of the number of peaks extracted

        for i, (mz_range, rt_range) in enumerate(zip(mz_ranges, rt_ranges)):
            mz_min, mz_max = mz_range
            rt_min, rt_max = rt_range

            # Filter spectra that fall into the rt range
            spectra_in_rt_range = spectra_table.filter(
                pc.and_(
                    pc.greater_equal(spectra_table['rt'], rt_min),
                    pc.less_equal(spectra_table['rt'], rt_max)
                )
            )

            # Skip if no spectra in this rt range
            if len(spectra_in_rt_range) == 0:
                continue

            # Convert to pandas DataFrame for easier handling
            spectra_df = spectra_in_rt_range.to_pandas()

            # Iterate over spectra in the rt range
            for idx, spectrum in spectra_df.iterrows():
                mz_array = np.array(spectrum['mz_array'])
                intensity_array = np.array(spectrum['intensity_array'])

                if len(mz_array) == 0:
                    continue

                # Filter peaks that fall into the m/z range
                mz_mask = (mz_array >= mz_min) & (mz_array <= mz_max)

                selected_mz = mz_array[mz_mask]
                selected_intensity = intensity_array[mz_mask]

                total_peaks_extracted += len(selected_mz)

                # You can store or process the selected peaks as needed
                # For benchmarking, we'll just count them

            # Optional: Print progress every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"MS{ms_level_label}: Processed {i + 1}/100 ranges...")

        end_time = time.time()
        print(f"MS{ms_level_label}: Extracted a total of {total_peaks_extracted} peaks from the m/z and rt ranges.")
        print(f"MS{ms_level_label}: Total time for extracting peaks from m/z and rt ranges: {end_time - start_time} seconds\n")

    # Perform extraction and benchmarking for MS1 spectra
    extract_peaks(ms1_spectra_table, ms1_min_rt, ms1_max_rt, ms_level_label=1)

    # Perform extraction and benchmarking for MS2 spectra
    extract_peaks(ms2_spectra_table, ms2_min_rt, ms2_max_rt, ms_level_label=2)


json_str = fillMetaDate(exp)
writeParquet(json_str)
benchmarkParquet()