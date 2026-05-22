#!/usr/bin/env python3
"""
Script to extract the first n entries from a JSON file and save them to a new file.

Usage:
    python extract_first_n_json_entries.py input.json
    python extract_first_n_json_entries.py input.json --n 50
"""

import json
import argparse
from pathlib import Path


def extract_first_n_entries(input_file, n=100):
    """
    Extracts the first n entries from a JSON file.
    
    Args:
        input_file: Path to the input JSON file
        n: Number of entries to extract (default: 100)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Fehler: Datei '{input_file}' nicht gefunden!")
        return
    
    print(f"Lade JSON-Datei: {input_path}")
    
    # JSON-Datei laden
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Typ überprüfen
    if isinstance(data, list):
       # total_entries = len(data)
        print(f"Gefunden: {total_entries} Einträge in der Liste")
        
        # Ersten n Einträge extrahieren
       ###### n_to_extract = min(n, total_entries)
        extracted_data = []
        
        print(f"Extrahiere die ersten {n_to_extract} Einträge...")
        for i in range(n_to_extract):
            extracted_data.append(data[i])
            
            # Fortschritt alle 5 Einträge ausgeben
            if (i + 1) % 5 == 0:
                print(f"  Fortschritt: {i + 1}/{n_to_extract} Einträge verarbeitet")
        
        # Finaler Fortschritt
        if n_to_extract % 5 != 0:
            print(f"  Fortschritt: {n_to_extract}/{n_to_extract} Einträge verarbeitet")
        
    elif isinstance(data, dict):
        # Wenn es ein Dictionary ist, die ersten n key-value Paare nehmen
        print(f"Gefunden: Dictionary mit {len(data)} Einträgen")
        
        keys = list(data.keys())[:n]
        n_to_extract = len(keys)
        extracted_data = {}
        
        print(f"Extrahiere die ersten {n_to_extract} Einträge...")
        for i, key in enumerate(keys):
            extracted_data[key] = data[key]
            
            # Fortschritt alle 5 Einträge ausgeben
            if (i + 1) % 5 == 0:
                print(f"  Fortschritt: {i + 1}/{n_to_extract} Einträge verarbeitet")
        
        # Finaler Fortschritt
        if n_to_extract % 5 != 0:
            print(f"  Fortschritt: {n_to_extract}/{n_to_extract} Einträge verarbeitet")
    else:
        print(f"Fehler: JSON-Datei hat unerwarteten Typ: {type(data)}")
        return
    
    # Output-Datei erstellen
    output_path = input_path.parent / f"{input_path.stem}_first_{n}_entries{input_path.suffix}"
    
    print(f"\nSpeichere Ergebnis in: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Fertig! {n_to_extract} Einträge erfolgreich gespeichert.")


def main():
    parser = argparse.ArgumentParser(
        description="Extrahiere die ersten n Einträge aus einer JSON-Datei"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Pfad zur Input-JSON-Datei'
    )
    parser.add_argument(
        '-n', '--n',
        type=int,
        default=100,
        help='Anzahl der zu extrahierenden Einträge (default: 100)'
    )
    
    args = parser.parse_args()
    
    extract_first_n_entries(args.input_file, args.n)


if __name__ == '__main__':
    main()
