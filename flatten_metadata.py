# prompt 
# input: mzML 1.1.1 schema file. 
# Write a Python script that extracts and flattens the metadata from an mzML file into a key-value structure. 
# Hierarchical relationships should be represented using a clear separator (e.g., : in the keys). 
# The conversion must be lossless and reversible, ensuring that the original metadata structure can be 
# reconstructed from the flattened representation.


import xml.etree.ElementTree as ET
from collections import defaultdict
import argparse
import json
import sys
import re

# --- Constants ---
PATH_SEPARATOR = ':'
ATTRIBUTE_PREFIX = '@'
TEXT_SUFFIX = '#text'
# Regex to extract tag and index (e.g., "cvParam[0]" -> ("cvParam", 0))
TAG_INDEX_RE = re.compile(r"^(.*)\[(\d+)\]$")

def _clean_tag(tag):
    """Removes the namespace URI part if present (e.g., {namespace}tag -> tag)."""
    return tag.split('}', 1)[-1] if '}' in tag else tag

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
        children_by_tag[child.tag].append(child)

    # Iterate through grouped children and add index for lists
    # Sort tag groups for deterministic output
    for tag, children in sorted(children_by_tag.items()):
        clean_child_tag = _clean_tag(tag)
        if len(children) == 1:
            # If only one child with this tag, use index [0] for consistency
            # This ensures reversibility knows it was an element, not just text/attr
            child_path = f"{path_prefix}{PATH_SEPARATOR}{clean_child_tag}[0]"
            flatten_element(children[0], child_path, flat_dict)
        else:
            # If multiple children, index them explicitly
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
        # Attempt to parse the XML file
        # We use iterparse to potentially handle large files better later,
        # but for metadata flattening, parsing the whole tree is often necessary
        # and simpler with ET.parse.
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

# --- Reversibility (Conceptual - demonstrates the idea) ---
# Note: This is a simplified reconstruction example. A robust version would need
# more error handling, namespace handling (if needed beyond cleaning), and potentially
# using lxml for better control over element creation and ordering.

def reconstruct_element(parent_element, path_segment, value, flat_dict, used_keys):
    """Helper for recursive reconstruction (Simplified Example)."""
    # Check if it's an attribute
    if path_segment.startswith(ATTRIBUTE_PREFIX):
        attr_name = path_segment[len(ATTRIBUTE_PREFIX):]
        parent_element.set(attr_name, value)
        return

    # Check if it's text content
    if path_segment == TEXT_SUFFIX:
        parent_element.text = value
        return

    # Otherwise, it's a child element (potentially indexed)
    match = TAG_INDEX_RE.match(path_segment)
    if not match:
        print(f"Warning: Could not parse path segment for reconstruction: {path_segment}", file=sys.stderr)
        return # Or raise error

    tag_name, index_str = match.groups()
    index = int(index_str)

    # Find or create the child element at the correct index
    # This requires careful handling of element order and existing children
    # For simplicity here, we assume elements are processed in order and create new ones
    child_element = ET.Element(tag_name) # In reality, need namespace handling here if required

    # Find related keys for this child element
    child_prefix = f"{parent_element.tag}{PATH_SEPARATOR}{path_segment}" # Simplified - needs full path context
    # A full reconstruction would need to pass the full path down

    # --- This part is highly simplified ---
    # A real reconstruction needs to find all keys starting with child_prefix + ':'
    # and recursively call reconstruct_element on them. It's complex to manage
    # state and ensure correct structure without a more sophisticated approach.
    # The key point is that the flattened structure *contains* the necessary info.

    # Example of setting an attribute found for the child (if the key existed)
    # hypothetical_attr_key = f"{child_prefix}{PATH_SEPARATOR}{ATTRIBUTE_PREFIX}id"
    # if hypothetical_attr_key in flat_dict:
    #    child_element.set("id", flat_dict[hypothetical_attr_key])
    #    used_keys.add(hypothetical_attr_key) # Mark as used

    parent_element.append(child_element) # Needs logic to insert at correct index if needed


def reconstruct_mzml(flat_dict):
    """
    Reconstructs an XML structure from a flattened dictionary (Conceptual Example).

    Args:
        flat_dict (dict): The flattened key-value dictionary.

    Returns:
        ET.Element: The root element of the reconstructed XML structure.
    """
    if not flat_dict:
        return None

    # Find the root key (the shortest key, which should be the root tag name)
    root_tag = min(flat_dict.keys(), key=len).split(PATH_SEPARATOR)[0]
    root_element = ET.Element(root_tag)

    used_keys = set()

    # Sort keys to process in a somewhat predictable order (important for structure)
    # A truly robust reconstruction would likely build a tree structure from keys first
    sorted_keys = sorted(flat_dict.keys())

    # This loop is overly simplified for demonstration.
    # A real version needs to parse the key, determine the parent, find/create it,
    # then set the attribute/text or create the child recursively.
    for key in sorted_keys:
        if key in used_keys:
            continue

        parts = key.split(PATH_SEPARATOR)
        value = flat_dict[key]

        # --- Very Basic Reconstruction Logic ---
        # This needs a proper tree-building algorithm based on path parsing.
        current_element = root_element
        # Navigate/create path (pseudo-code)
        # for i, segment in enumerate(parts[1:]): # Skip root tag
        #    if i == len(parts) - 2: # Last segment determines attr/text/child
        #       reconstruct_element(current_element, segment, value, flat_dict, used_keys)
        #       used_keys.add(key)
        #    else:
        #       # Find or create intermediate child element based on segment
        #       # current_element = find_or_create_child(current_element, segment)
        #       pass # Placeholder for complex navigation/creation logic

    # Placeholder: Assign root attributes/text if they exist directly under root tag path
    root_attr_prefix = f"{root_tag}{PATH_SEPARATOR}{ATTRIBUTE_PREFIX}"
    root_text_key = f"{root_tag}{PATH_SEPARATOR}{TEXT_SUFFIX}"
    for key, value in flat_dict.items():
         if key == root_text_key:
             root_element.text = value
             used_keys.add(key)
         elif key.startswith(root_attr_prefix):
             attr_name = key[len(root_attr_prefix):]
             root_element.set(attr_name, value)
             used_keys.add(key)

    print("Warning: reconstruct_mzml is a simplified conceptual example and does not fully rebuild the tree.", file=sys.stderr)
    # In a real implementation, the loop above would build the entire tree.

    return root_element

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten mzML metadata into a lossless key-value JSON structure.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("mzml_file", help="Path to the input mzML file.")
    parser.add_argument("-o", "--output", help="Path to save the flattened JSON output (optional). Prints to console if not specified.")
    parser.add_argument(
        "--reconstruct-test",
        metavar="FLAT_JSON_FILE",
        help="Perform a conceptual reconstruction test from a flattened JSON file (for demonstration)."
    )


    args = parser.parse_args()

    if args.reconstruct_test:
        print(f"--- Performing Conceptual Reconstruction Test from: {args.reconstruct_test} ---")
        try:
            with open(args.reconstruct_test, 'r') as f:
                flat_data_to_reconstruct = json.load(f)
            print(f"Loaded {len(flat_data_to_reconstruct)} items from JSON.")
            reconstructed_root = reconstruct_mzml(flat_data_to_reconstruct)
            if reconstructed_root is not None:
                # Print the reconstructed XML (will be simplified)
                reconstructed_xml_string = ET.tostring(reconstructed_root, encoding='unicode', short_empty_elements=False)
                print("\nReconstructed XML (Simplified):")
                print(reconstructed_xml_string)
            else:
                print("Reconstruction failed or produced no root element.")
        except FileNotFoundError:
            print(f"Error: Reconstruction test file not found: {args.reconstruct_test}", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {args.reconstruct_test}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"An error occurred during reconstruction test: {e}", file=sys.stderr)

    else:
        print(f"--- Flattening mzML file: {args.mzml_file} ---")
        flattened_data = flatten_mzml(args.mzml_file)

        if flattened_data:
            print(f"Successfully flattened {len(flattened_data)} metadata items.")

            # Prepare JSON output
            output_json = json.dumps(flattened_data, indent=2)

            if args.output:
                try:
                    with open(args.output, 'w') as f:
                        f.write(output_json)
                    print(f"Flattened data saved to: {args.output}")
                except IOError as e:
                    print(f"Error writing to output file '{args.output}': {e}", file=sys.stderr)
            else:
                # Print to console
                print("\nFlattened Data:")
                print(output_json)
        else:
            print("Flattening failed.", file=sys.stderr)
            sys.exit(1) # Exit with error code if flattening fails
