# -*- coding: utf-8 -*-
"""
Script to flatten mzML metadata into a lossless key-value JSON structure
and reconstruct the mzML XML from the flattened JSON.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
import argparse
import json
import sys
import re
import io  # For string conversion during XML output

# --- Constants ---
PATH_SEPARATOR = ':'
ATTRIBUTE_PREFIX = '@'
TEXT_SUFFIX = '#text'
# Regex to extract tag and index (e.g., "cvParam[0]" -> ("cvParam", 0))
TAG_INDEX_RE = re.compile(r"^(.*)\[(\d+)\]$")
# Default mzML namespace (adjust if your flattening preserved others)
DEFAULT_MZML_NAMESPACE = "http://psi.hupo.org/ms/mzml"
NAMESPACE_PREFIX = "" # The prefix used for the mzML namespace


# --- Helper Functions ---

def _clean_tag(tag):
    """Removes the namespace URI part if present (e.g., {namespace}tag -> tag)."""
    return tag.split('}', 1)[-1] if '}' in tag else tag


# --- Flattening Logic ---

def flatten_element(element, path_prefix, flat_dict):
    """
    Recursively flattens an XML element and its children into a dictionary.

    Args:
        element (ET.Element): The current XML element to process.
        path_prefix (str): The key prefix representing the path to this element.
        flat_dict (dict): The dictionary to store the flattened key-value pairs.
    """
    # 1. Process Attributes
    # Sort attributes for deterministic output (good practice)
    for name, value in sorted(element.attrib.items()):
        # Store attribute name without namespace prefix for simplicity in keys,
        # assuming no clashes. Reconstruction adds default namespace back.
        attr_key = f"{path_prefix}{PATH_SEPARATOR}{ATTRIBUTE_PREFIX}{_clean_tag(name)}"
        flat_dict[attr_key] = value

    # 2. Process Text Content (only if it contains non-whitespace characters)
    if element.text and element.text.strip():
        text_key = f"{path_prefix}{PATH_SEPARATOR}{TEXT_SUFFIX}"
        flat_dict[text_key] = element.text.strip()

    # 3. Process Children
    # Group children by tag to handle lists/multiple occurrences correctly
    children_by_tag = defaultdict(list)
    for child in element:
        # Use the full tag (with namespace URI) for grouping
        children_by_tag[child.tag].append(child)

    # Iterate through grouped children and add index for lists
    # Sort tag groups for deterministic output
    for tag, children in sorted(children_by_tag.items()):
        clean_child_tag = _clean_tag(tag) # Use clean tag for the key path
        # Always use index for elements to ensure reversibility knows it's an element
        for index, child_element in enumerate(children):
            child_path = f"{path_prefix}{PATH_SEPARATOR}{clean_child_tag}[{index}]"
            flatten_element(child_element, child_path, flat_dict)

def flatten_mzml(mzml_filepath):
    """
    Parses an mzML file and flattens its structure into a key-value dictionary.

    Args:
        mzml_filepath (str): The path to the mzML file.

    Returns:
        dict: A dictionary representing the flattened mzML metadata.
              Returns None if the file cannot be parsed.
    """
    try:
        # Use ET.parse for simplicity in accessing the whole tree needed for structure.
        # For very large files where only specific data is needed, iterparse is better,
        # but flattening requires the full hierarchy.
        tree = ET.parse(mzml_filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file '{mzml_filepath}': {e}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"Error: File not found at '{mzml_filepath}'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}", file=sys.stderr)
        return None

    flat_metadata = {}
    # Start flattening from the root element, using its cleaned tag as the initial path
    initial_path = _clean_tag(root.tag)
    flatten_element(root, initial_path, flat_metadata)
    return flat_metadata


# --- Reconstruction Logic ---

def build_intermediate_structure(flat_dict):
    """
    Parses the flat dictionary keys to build a hierarchical intermediate structure
    (nested dictionaries and lists) suitable for reconstructing the XML.

    Args:
        flat_dict (dict): The flattened key-value dictionary.

    Returns:
        dict: A nested dictionary/list structure representing the XML hierarchy.
              Returns None if input is empty or invalid root is detected.
    """
    if not flat_dict:
        return None

    intermediate_tree = {}
    root_tag_candidates = set()

    # Sort keys to process parents before children where possible (helps structure build)
    sorted_keys = sorted(flat_dict.keys())

    for key in sorted_keys:
        value = flat_dict[key]
        parts = key.split(PATH_SEPARATOR)
        if not parts:
            print(f"Warning: Skipping empty key: '{key}'", file=sys.stderr)
            continue

        root_tag_candidates.add(parts[0])
        current_level = intermediate_tree
        path_traversed = [] # Keep track for error messages

        for i, part in enumerate(parts):
            path_traversed.append(part)
            is_last_part = (i == len(parts) - 1)

            # --- Handle Attributes or Text on the *current* target element ---
            if is_last_part:
                if part.startswith(ATTRIBUTE_PREFIX):
                    attr_name = part[len(ATTRIBUTE_PREFIX):]
                    # Attributes belong to the parent element represented by current_level
                    if '@attributes' not in current_level:
                        current_level['@attributes'] = {}
                    current_level['@attributes'][attr_name] = value
                elif part == TEXT_SUFFIX:
                    # Text belongs to the parent element represented by current_level
                    current_level['#text'] = value
                else:
                    # This case handles elements that might have *only* appeared
                    # as the last part of a key (e.g., an empty element defined
                    # only by its path, like "root:child[0]").
                    # We ensure the element exists in the structure.
                    match = TAG_INDEX_RE.match(part)
                    if not match:
                         # If it's the very first part (root), it's okay
                         if i == 0:
                             tag_name = part
                             if tag_name not in current_level:
                                 current_level[tag_name] = {} # Create root dict if not exists
                             # No value assignment here, just ensure structure exists
                         else:
                             print(f"Warning: Malformed final segment '{part}' in key '{key}'. Expected attribute, text, or indexed element. Skipping.", file=sys.stderr)
                             break # Stop processing this key
                    else:
                        tag_name, index_str = match.groups()
                        index = int(index_str)

                        if tag_name not in current_level:
                            current_level[tag_name] = [] # Initialize list for the tag

                        # Ensure list is long enough
                        while len(current_level[tag_name]) <= index:
                            current_level[tag_name].append({}) # Add empty dict placeholders

                        # Ensure the target index holds a dictionary
                        if not isinstance(current_level[tag_name][index], dict):
                             current_level[tag_name][index] = {}
                        # No value assignment here, attributes/text handled above

                break # Done processing this key

            # --- Handle Element Path Step (Not the last part) ---
            match = TAG_INDEX_RE.match(part)
            if not match:
                 # This should only be the root element if i == 0
                if i == 0:
                    tag_name = part
                    if tag_name not in current_level:
                         current_level[tag_name] = {} # Root is a single dict initially
                    current_level = current_level[tag_name] # Move into the root dict
                else:
                    # This case shouldn't happen if flattening includes indices correctly
                    print(f"Warning: Malformed intermediate segment '{part}' in key '{key}'. Expected indexed element. Skipping.", file=sys.stderr)
                    break # Stop processing this key
            else:
                # It's an indexed element path part
                tag_name, index_str = match.groups()
                index = int(index_str)

                # Ensure the list for this tag exists at the current level
                if tag_name not in current_level:
                    current_level[tag_name] = []

                # Ensure the list is long enough to accommodate the index
                while len(current_level[tag_name]) <= index:
                    current_level[tag_name].append({}) # Add empty dict placeholders

                # Ensure the target index holds a dictionary (might have been placeholder)
                if not isinstance(current_level[tag_name][index], dict):
                     current_level[tag_name][index] = {}

                # Move to the next level (the dictionary at the specific index)
                current_level = current_level[tag_name][index]

    # --- Validation and Root Extraction ---
    if len(root_tag_candidates) == 0:
        print("Error: No data found in flattened dictionary.", file=sys.stderr)
        return None
    # Finding the actual root from the built structure
    actual_root_tags = [k for k in intermediate_tree if not k.startswith('@') and k != '#text']

    if len(actual_root_tags) != 1:
         # Try using candidates if structure seems problematic
         possible_roots = [r for r in root_tag_candidates if r in intermediate_tree]
         if len(possible_roots) == 1:
             root_tag = possible_roots[0]
             print(f"Warning: Multiple potential roots or structure issues detected. Using determined root: '{root_tag}'", file=sys.stderr)
             return {root_tag: intermediate_tree[root_tag]}
         else:
            print(f"Error: Could not definitively determine single root element. Found possible: {actual_root_tags} from structure, candidates: {root_tag_candidates}", file=sys.stderr)
            return None

    root_tag = actual_root_tags[0]
    # Return the structure wrapped in the root tag key
    return {root_tag: intermediate_tree[root_tag]}


def create_element_from_intermediate(tag_name, node_data, namespace_uri):
    """
    Recursively creates an ET.Element from the intermediate structure node.

    Args:
        tag_name (str): The tag name for the element to create.
        node_data (dict): The dictionary representing this element's data
                          (attributes, text, children) from the intermediate structure.
        namespace_uri (str): The XML namespace URI to apply to created elements.
                             If empty or None, no namespace is added.

    Returns:
        ET.Element: The constructed XML element.
    """
    # Prepend namespace URI if provided and not empty
    full_tag = f"{{{namespace_uri}}}{tag_name}" if namespace_uri else tag_name
    element = ET.Element(full_tag)

    # Set attributes
    if '@attributes' in node_data:
        # Sort attributes for deterministic output
        for attr_name, attr_value in sorted(node_data['@attributes'].items()):
            # Note: Attribute namespaces are not handled simply here.
            # We assume attributes are not namespace-qualified in the flattened keys.
            # If they were, attr_name would need parsing/handling.
            element.set(attr_name, attr_value)

    # Set text content
    if '#text' in node_data:
        element.text = node_data['#text']

    # Recursively create and append child elements
    # Extract child tags (keys not starting with @ and not #text)
    child_keys = sorted([k for k in node_data if not k.startswith('@') and k != '#text'])

    for child_tag in child_keys:
        child_list_or_dict = node_data[child_tag]

        if isinstance(child_list_or_dict, list):
            # Handle lists of children (multiple elements with same tag)
            for child_node_data in child_list_or_dict:
                if isinstance(child_node_data, dict): # Ensure it's valid data
                    child_element = create_element_from_intermediate(
                        child_tag, child_node_data, namespace_uri
                    )
                    element.append(child_element)
                # else: Optional: print warning about non-dict item in list?
        elif isinstance(child_list_or_dict, dict):
             # Handle single child represented directly as dict (shouldn't happen with current intermediate builder)
             print(f"Warning: Found dictionary directly under tag '{child_tag}' instead of list. Processing as single child.", file=sys.stderr)
             child_element = create_element_from_intermediate(
                 child_tag, child_list_or_dict, namespace_uri
             )
             element.append(child_element)
        # else: Optional: print warning about unexpected child data type?

    return element

def reconstruct_mzml(flat_dict, namespace_uri=DEFAULT_MZML_NAMESPACE):
    """
    Reconstructs an XML structure (ElementTree root) from a flattened dictionary.

    Args:
        flat_dict (dict): The flattened key-value dictionary.
        namespace_uri (str): The XML namespace to apply to elements.
                               Defaults to the standard mzML namespace.
                               Pass None or "" to omit namespaces.

    Returns:
        ET.Element: The root element of the reconstructed XML structure, or None on failure.
    """
    print("Building intermediate structure from flattened data...")
    intermediate = build_intermediate_structure(flat_dict)

    if not intermediate:
        print("Failed to build intermediate structure.", file=sys.stderr)
        return None

    # Extract the single root tag and its data dictionary
    root_tag = list(intermediate.keys())[0]
    root_data = intermediate[root_tag]

    print(f"Reconstructing XML tree from root '{root_tag}'...")

    # Register namespace for potentially cleaner output XML
    effective_ns_uri = namespace_uri if namespace_uri else ""
    if effective_ns_uri and NAMESPACE_PREFIX:
       try:
           ET.register_namespace(NAMESPACE_PREFIX, effective_ns_uri)
           print(f"Registered namespace prefix '{NAMESPACE_PREFIX}' for URI '{effective_ns_uri}'")
       except ValueError as e:
           print(f"Warning: Could not register namespace prefix '{NAMESPACE_PREFIX}': {e}. Output might use full URIs.", file=sys.stderr)
       except Exception as e: # Catch other potential issues during registration
           print(f"Warning: An unexpected error occurred during namespace registration: {e}", file=sys.stderr)


    # Start the recursive element creation process
    root_element = create_element_from_intermediate(root_tag, root_data, effective_ns_uri)
    print("XML tree reconstruction complete.")
    return root_element


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten mzML metadata to JSON or reconstruct mzML XML from JSON.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Action to perform')

    # --- Flatten Sub-command ---
    parser_flatten = subparsers.add_parser('flatten', help='Flatten an mzML file to JSON')
    parser_flatten.add_argument("mzml_file", help="Path to the input mzML file.")
    parser_flatten.add_argument("-o", "--output",
                                help="Path to save the flattened JSON output. Prints to console if omitted.")

    # --- Reconstruct Sub-command ---
    parser_reconstruct = subparsers.add_parser('reconstruct', help='Reconstruct mzML XML from flattened JSON')
    parser_reconstruct.add_argument("flat_json_file", help="Path to the input flattened JSON file.")
    parser_reconstruct.add_argument("-o", "--output",
                                    help="Path to save the reconstructed XML output. Prints to console if omitted.")
    parser_reconstruct.add_argument("--namespace-uri", default=DEFAULT_MZML_NAMESPACE,
                                    help=f"XML namespace URI for reconstruction (default: {DEFAULT_MZML_NAMESPACE}). Use empty string '' to disable.")
    parser_reconstruct.add_argument("--pretty", action='store_true',
                                    help="Pretty-print the output XML with indentation (requires Python 3.9+).")

    args = parser.parse_args()

    # --- Execute Flatten Command ---
    if args.command == 'flatten':
        print(f"--- Flattening mzML file: {args.mzml_file} ---")
        flattened_data = flatten_mzml(args.mzml_file)

        if flattened_data:
            print(f"Successfully flattened {len(flattened_data)} metadata items.")
            # Ensure separators etc are standard ASCII for JSON compatibility
            output_json = json.dumps(flattened_data, indent=2, ensure_ascii=True)

            if args.output:
                try:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(output_json)
                    print(f"Flattened data saved to: {args.output}")
                except IOError as e:
                    print(f"Error writing to output file '{args.output}': {e}", file=sys.stderr)
                    sys.exit(1)
            else:
                print("\nFlattened Data:")
                print(output_json)
        else:
            print("Flattening failed.", file=sys.stderr)
            sys.exit(1)

    # --- Execute Reconstruct Command ---
    elif args.command == 'reconstruct':
        print(f"--- Reconstructing XML from: {args.flat_json_file} ---")
        try:
            with open(args.flat_json_file, 'r', encoding='utf-8') as f:
                flat_data_to_reconstruct = json.load(f)
            print(f"Loaded {len(flat_data_to_reconstruct)} flattened items from JSON.")

            # Determine namespace URI to use (handle empty string for disabling)
            ns_uri = args.namespace_uri if args.namespace_uri else ""

            reconstructed_root = reconstruct_mzml(flat_data_to_reconstruct, ns_uri)

            if reconstructed_root is not None:
                # Create ElementTree for writing
                tree = ET.ElementTree(reconstructed_root)

                # Optional Pretty Printing (Python 3.9+)
                if args.pretty:
                    if hasattr(ET, 'indent'):
                        print("Applying XML indentation (pretty-printing)...")
                        ET.indent(tree, space="  ", level=0)
                    else:
                        print("Warning: XML pretty-printing (ET.indent) requires Python 3.9+. Outputting compact XML.", file=sys.stderr)

                # Prepare XML string using BytesIO for reliable encoding
                xml_bytes_io = io.BytesIO()
                tree.write(xml_bytes_io, encoding='utf-8', xml_declaration=True)
                reconstructed_xml_string = xml_bytes_io.getvalue().decode('utf-8')

                # Output to file or console
                if args.output:
                    try:
                        with open(args.output, 'w', encoding='utf-8') as f:
                            f.write(reconstructed_xml_string)
                        print(f"Reconstructed XML saved to: {args.output}")
                    except IOError as e:
                        print(f"Error writing to output file '{args.output}': {e}", file=sys.stderr)
                        sys.exit(1)
                else:
                    print("\nReconstructed XML:")
                    # Ensure print handles unicode correctly, especially on Windows
                    # sys.stdout.reconfigure(encoding='utf-8') # Python 3.7+
                    print(reconstructed_xml_string)

            else:
                print("XML reconstruction failed to produce a root element.", file=sys.stderr)
                sys.exit(1)

        except FileNotFoundError:
            print(f"Error: Reconstruction input file not found: {args.flat_json_file}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from '{args.flat_json_file}': {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during reconstruction: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)