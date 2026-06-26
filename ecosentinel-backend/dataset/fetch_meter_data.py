import os
import csv
import pandas as pd
import requests

# 1. Map your OBIS codes to their canonical (real) names here
OBIS_MAP = {
    "1.0.31.27.0.255": "Phase_1_Current_A",
    "1.0.51.27.0.255": "Phase_2_Current_A",
    "1.0.71.27.0.255": "Phase_3_Current_A",
    "1.0.32.27.0.255": "Phase_1_Voltage_V",
    "1.0.52.27.0.255": "Phase_2_Voltage_V",
    "1.0.72.27.0.255": "Phase_3_Voltage_V",
    "1.0.1.29.0.255": "Active_Energy_Import_Wh",
    "1.0.2.29.0.255": "Active_Energy_Export_Wh",
    "1.0.5.29.0.255": "Reactive_Energy_QI_VArh",
    "1.0.6.29.0.255": "Reactive_Energy_QII_VArh",
    "1.0.7.29.0.255": "Reactive_Energy_QIII_VArh",
    "1.0.8.29.0.255": "Reactive_Energy_QIV_VArh",
    "1.0.9.29.0.255": "Apparent_Energy_Import_VAh",
    "1.0.10.29.0.255": "Apparent_Energy_Export_VAh",
    "0.1.96.12.5.255": "Cellular_Signal_Strength_CSQ",
    "1.0.12.27.0.255": "Average_Voltage_V",
    "1.0.11.27.0.255": "Average_Current_A",
}

# Base configuration
API_URL = "http://[2406:da1a:536:b201:f8fc:9161:b808:d5e2]:3005/api/api/MeterData/meterobis"
OUTPUT_DIR = "dataset/Test_Dataset"  # Folder where all individual CSVs will be stored

# ----------------- EXCEL INPUT CONFIGURATION -----------------
EXCEL_FILE = "dataset/Meter Management - EcoSEnter.xlsx"
METER_SERIAL_COLUMN = "Serial No"
# -------------------------------------------------------------


def parse_raw_value(raw_value_str):
    """Parses the pipe-delimited rawValue string into a timestamp and a dictionary

    of mapped OBIS canonical names to their values.
    """
    if not raw_value_str:
        return None, {}

    parts = raw_value_str.split("|")
    if not parts:
        return None, {}

    # Extract timestamp from the first element (e.g., "1,0.0.1.0.0.255,2,2026-06-10 15:30:00,")
    first_part = parts[0].split(",")
    # The 4th item (index 3) holds the reading timestamp string
    timestamp = first_part[3] if len(first_part) > 3 else "Unknown"

    data_points = {}
    # Parse the remaining OBIS components
    for part in parts[1:]:
        tokens = part.split(",")
        if len(tokens) >= 4:
            obis_code = tokens[1]
            value = tokens[3]

            # Convert OBIS code to canonical name if found in the map
            canonical_name = OBIS_MAP.get(obis_code, f"OBIS_{obis_code}")
            data_points[canonical_name] = value

    return timestamp, data_points


def load_meter_metadata_from_excel(file_path, serial_col):
    """Reads Excel and builds a mapping dictionary containing metadata for each meter."""
    metadata_map = {}
    try:
        print(f"Reading meter serials and location details from Excel: {file_path}...")
        
        # header=1 to read from the second row containing actual headers
        df = pd.read_excel(file_path, header=1)
        df.columns = df.columns.str.strip()

        if serial_col not in df.columns:
            raise ValueError(f"Column '{serial_col}' not found in Excel sheet. Available: {list(df.columns)}")

        # Fill empty string for text columns so we don't write "nan" to our CSV files
        text_cols = ["Phase Type", "Location", "Address"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()
            else:
                df[col] = ""  # Default empty if the column is missing in the sheet

        # Iterate through Excel rows to build metadata profiles
        for _, row in df.iterrows():
            serial = str(row[serial_col]).strip()
            if not serial or serial == "nan":
                continue
                
            metadata_map[serial] = {
                "phase_type": row["Phase Type"],
                "location": row["Location"],
                "address": row["Address"]
            }
            
        print(f"Successfully loaded metadata for {len(metadata_map)} unique meters.")
        return metadata_map
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return {}


def main():
    # Make sure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load complete metadata tracking object map from excel
    meta_map = load_meter_metadata_from_excel(EXCEL_FILE, METER_SERIAL_COLUMN)

    if not meta_map:
        print("No meter configurations found to process. Exiting.")
        return

    headers_api = {"accept": "text/plain"}
    print("Starting data fetch from HES endpoint...")

    for serial, meta in meta_map.items():
        params = {"obisCode": "1.0.99.1.0.255", "meterSerial": serial}
        meter_records = []
        dynamic_headers = set()

        try:
            response = requests.get(API_URL, params=params, headers=headers_api, timeout=15)
            response.raise_for_status()
            data = response.json()

            if not data:
                print(f"No data returned for meter: {serial} ({meta['location']})")
                continue

            for record in data:
                raw_val = record.get("rawValue", "")
                timestamp, metrics = parse_raw_value(raw_val)

                # Base record mapping + Location tracking parameters injected here
                row = {
                    "meter_serial": serial,
                    "phase_type": meta["phase_type"],
                    "location": meta["location"],
                    "address": meta["address"],
                    "timestamp": timestamp
                }

                # Merge telemetry metrics into the row map
                row.update(metrics)
                meter_records.append(row)

                # Track captured parameters dynamically
                dynamic_headers.update(metrics.keys())

            # Write out individual CSV files per meter serial if data exists
            if meter_records:
                sorted_metrics = sorted(list(dynamic_headers))
                # New standard row header footprint order
                fieldnames = ["meter_serial", "phase_type", "location", "address", "timestamp"] + sorted_metrics
                output_csv_path = os.path.join(OUTPUT_DIR, f"{serial}.csv")

                with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    for record in meter_records:
                        writer.writerow({col: record.get(col, "") for col in fieldnames})

                print(f"-> Successfully saved: {output_csv_path} | Loc: {meta['location']}")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for meter {serial}: {e}")
        except ValueError:
            print(f"Failed to parse JSON response for meter {serial}")

    print("\nAll processing completed!")


if __name__ == "__main__":
    main()