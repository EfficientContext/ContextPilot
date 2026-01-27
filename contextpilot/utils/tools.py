import json
from tqdm import tqdm


def combine_jsonl_files(file1_path, file2_path, output_path):
    """
    Combine two JSONL files based on matching qid and question_id
    
    Args:
        file1_path: Path to file with {"qid": xxx, "question": xxxx, "answers": xxxx}
        file2_path: Path to file with {"question_id": xxx, "topk_ids": xxx}
        output_path: Path for the combined output file
    """
    
    # Read first file and create a dictionary with qid as key
    qid_data = {}
    with open(file1_path, 'r', encoding='utf-8') as f1:
        for line in f1:
            data = json.loads(line.strip())
            qid_data[data['qid']] = data
    
    # Read second file and combine with first file data
    combined_data = []
    with open(file2_path, 'r', encoding='utf-8') as f2:
        for line in f2:
            data = json.loads(line.strip())
            question_id = data['question_id']
            
            # If we find a matching qid in the first file
            if question_id in qid_data:
                # Combine the data from both files
                combined_record = {**qid_data[question_id], **data}
                # Remove duplicate id field (keeping qid, removing question_id)
                del combined_record['question_id']
                
                combined_data.append(combined_record)
            else:
                print(f"Warning: No matching qid found for question_id: {question_id}")
    
    # Write combined data to output file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for record in combined_data:
            output_file.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Combined {len(combined_data)} records and saved to {output_path}")

import json
from typing import List, Dict, Any

def read_jsonl_file(file_path: str) -> List[Dict[Any, Any]]:
    """
    Read a JSONL file and return a list of dictionaries.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries from the JSONL file
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON on line {line_number}: {e}")
                        print(f"Problematic line: {line}")
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return []

def reorder_jsonl_data(data: List[Dict[Any, Any]], reordered_indices: List[int]) -> List[Dict[Any, Any]]:
    """
    Reorder the list of dictionaries based on the given indices.
    
    Args:
        data: List of dictionaries from JSONL file
        reordered_indices: List of indices indicating the new order
        
    Returns:
        Reordered list of dictionaries
        
    Example:
        If original data is [item0, item1, item2] and reordered_indices is [2, 0, 1],
        the result will be [item2, item0, item1]
    """
    if not data:
        print("Warning: Data list is empty.")
        return []
    
    if not reordered_indices:
        print("Warning: Reordered indices list is empty.")
        return data
    
    # Validate indices
    max_index = len(data) - 1
    invalid_indices = [idx for idx in reordered_indices if idx < 0 or idx > max_index]
    
    if invalid_indices:
        print(f"Error: Invalid indices found: {invalid_indices}. Data has {len(data)} items (indices 0-{max_index}).")
        return data
    
    if len(reordered_indices) != len(data):
        print(f"Warning: Length mismatch. Data has {len(data)} items but {len(reordered_indices)} indices provided.")
        print("Using available indices only.")
    
    # Reorder the data
    try:
        reordered_data = []
        used_indices = set()
        
        for idx in reordered_indices:
            if idx in used_indices:
                print(f"Warning: Duplicate index {idx} found. Skipping duplicate.")
                continue
            if 0 <= idx < len(data):
                reordered_data.append(data[idx])
                used_indices.add(idx)
            else:
                print(f"Warning: Index {idx} is out of range. Skipping.")
        
        return reordered_data
        
    except Exception as e:
        print(f"Error during reordering: {e}")
        return data

def write_jsonl_file(data: List[Dict[Any, Any]], output_path: str) -> bool:
    """
    Write a list of dictionaries to a JSONL file.
    
    Args:
        data: List of dictionaries to write
        output_path: Path for the output JSONL file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)
                file.write(json_line + '\n')
        return True
    except Exception as e:
        print(f"Error writing to file '{output_path}': {e}")
        return False

def read_and_reorder_jsonl(input_file_path: str, reordered_indices: List[int], output_file_path: str = None) -> List[Dict[Any, Any]]:
    """
    Complete function to read JSONL, reorder based on indices, and optionally save to new file.
    
    Args:
        input_file_path: Path to the input JSONL file
        reordered_indices: List of indices for reordering
        output_file_path: Optional path to save reordered data
        
    Returns:
        Reordered list of dictionaries
    """
    print(f"Reading JSONL file: {input_file_path}")
    
    # Read the JSONL file
    data = read_jsonl_file(input_file_path)
    
    if not data:
        print("No data to reorder.")
        return []
    
    print(f"Successfully read {len(data)} items from JSONL file.")
    
    # Reorder the data
    print(f"Reordering data based on {len(reordered_indices)} indices...")
    reordered_data = reorder_jsonl_data(data, reordered_indices)
    
    print(f"Reordering complete. Result has {len(reordered_data)} items.")
    
    # Optionally save to output file
    if output_file_path:
        print(f"Saving reordered data to: {output_file_path}")
        if write_jsonl_file(reordered_data, output_file_path):
            print("Successfully saved reordered data.")
        else:
            print("Failed to save reordered data.")
    
    return reordered_data

def split_to_other_topk(k, jsonl_file, output_file):
    """
    Split the top-k results from a JSON file into separate files for each k value.
    
    Args:
        k: The number of top results to split
        json_file: Path to the input JSON file containing top-k results
    """
    import os
    
    if not os.path.exists(jsonl_file):
        print(f"Error: File '{jsonl_file}' does not exist.")
        return
    
    with open(jsonl_file, 'r', encoding='utf-8') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                data = json.loads(line.strip())
                if 'top_k_doc_id' in data and isinstance(data['top_k_doc_id'], list):
                    topk_ids = data['top_k_doc_id'][:k]        
                text = data['text']
                answer = data['answer']
                qid = data['qid']

                new_data = {
                    "qid": qid,
                    "text": text,
                    "answer": answer,
                    "top_k_doc_id": topk_ids
                }
                outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

# def analyze_batch_efficiency(batch_results):
#     """
#     Analyze the efficiency of the batching system.
    
#     Args:
#         batch_results: Results from batch_queries_by_groups
    
#     Returns:
#         dict: Analysis metrics
#     """
#     first_batch = batch_results['first_batch']
#     second_batch = batch_results['second_batch']
#     groups = batch_results['groups']
    
#     analysis = {
#         'first_batch_size': len(first_batch),
#         'second_batch_size': len(second_batch),
#         'total_groups': len(groups),
#         'compression_ratio': batch_results['compression_ratio'],
#         'batch_size_ratio': len(first_batch) / (len(first_batch) + len(second_batch)) if (len(first_batch) + len(second_batch)) > 0 else 0,
#         'average_group_size': sum(len(group) for group in groups) / len(groups) if groups else 0,
#         'largest_group_size': max(len(group) for group in groups) if groups else 0,
#         'smallest_group_size': min(len(group) for group in groups) if groups else 0,
#     }
    
#     # Calculate potential prefix sharing within first batch
#     if len(first_batch) > 1:
#         total_prefix_sharing = 0
#         comparisons = 0
#         for i in range(len(first_batch)):
#             for j in range(i + 1, len(first_batch)):
#                 total_prefix_sharing += longest_common_prefix_length(first_batch[i], first_batch[j])
#                 comparisons += 1
#         analysis['avg_prefix_sharing_first_batch'] = total_prefix_sharing / comparisons if comparisons > 0 else 0
#     else:
#         analysis['avg_prefix_sharing_first_batch'] = 0
    
#     return analysis

def chunk_documents(input_data, chunk_size=1000, chunk_overlap=200, out_file="nodes.jsonl"):
    """
    Chunk documents from input data containing title and text.
    
    Args:
        input_data: List of dictionaries with 'title' and 'text' keys, or
                   single dictionary with 'title' and 'text' keys
        chunk_size: Size of each chunk in tokens/words
        chunk_overlap: Overlap between consecutive chunks
        out_file: Output file path for JSONL format
    """
    if not isinstance(input_data, list):
        input_data = [input_data]

    all_chunks = []
    chunk_id_counter = 0
    for doc_id_counter, doc in enumerate(tqdm.tqdm(input_data, desc="Chunking documents")):
        title = doc.get("title", f"doc_{doc_id_counter}")
        text = doc.get("text", "")

        if not text:
            continue

        # Using simple space splitting for words. For more advanced tokenization,
        # a library like tiktoken or transformers would be needed.
        words = text.split()
        
        start_index = 0
        
        # Calculate step size, ensure it's positive
        step = chunk_size - chunk_overlap
        if step <= 0:
            # To prevent infinite loops, if overlap is >= chunk_size,
            # we process chunks sequentially without overlap.
            step = chunk_size

        while start_index < len(words):
            end_index = start_index + chunk_size
            chunk_words = words[start_index:end_index]
            chunk_text = " ".join(chunk_words)
            
            chunk_node = {
                "chunk_id": chunk_id_counter,
                "text": chunk_text,
                "title": title,
            }
            all_chunks.append(chunk_node)
            
            chunk_id_counter += 1
            start_index += step

    return all_chunks