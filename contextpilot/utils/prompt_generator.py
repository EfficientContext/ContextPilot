from typing import List, Dict, Any, Optional, Tuple
import warnings
import os

try:
    # Suppress transformers warning about PyTorch not being installed
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    AutoTokenizer = None
    HAS_TRANSFORMERS = False

# Global tokenizer instance (cached)
_tokenizer = None
_tokenizer_model_name = None


def get_tokenizer(model_name: str):
    """Get or create a tokenizer instance for the given model."""
    global _tokenizer, _tokenizer_model_name
    
    if not HAS_TRANSFORMERS:
        return None
    
    if _tokenizer is not None and _tokenizer_model_name == model_name:
        return _tokenizer
    
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _tokenizer_model_name = model_name
        return _tokenizer
    except Exception as e:
        print(f"Warning: Failed to load tokenizer for {model_name}: {e}")
        return None


def apply_chat_template(
    prompt: str,
    tokenizer=None,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    add_generation_prompt: bool = True
) -> str:
    """
    Apply chat template to a prompt.
    
    Args:
        prompt: The user's message/prompt text
        tokenizer: Pre-loaded tokenizer instance (optional)
        model_name: Model name to load tokenizer from (if tokenizer not provided)
        system_prompt: Optional system prompt to prepend
        add_generation_prompt: Whether to add generation prompt (default True)
    
    Returns:
        The formatted prompt string with chat template applied,
        or the original prompt if no tokenizer is available.
    """
    # Get tokenizer
    if tokenizer is None and model_name:
        tokenizer = get_tokenizer(model_name)
    
    if tokenizer is None:
        return prompt
    
    # Build messages list
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return formatted
    except Exception as e:
        print(f"Warning: Failed to apply chat template: {e}. Using raw prompt.")
        return prompt


PROMPT_TEMPLATE = '''
**Note**:
1. Please answer within 10 words or fewer. If the answer is Yes or No, just say Yes or No
2. Do not include any special characters like <answer> or <answer/>."

With provided related documents:
<documents_section>
{docs_section}
</documents_section>

Answer the question:
<question_section>
{question}
</question_section>

Please read the documents in the following ranking and answer the question:
<importance_ranking_section>
{importance_ranking}
</importance_ranking_section>

Please prioritize information from higher-ranked documents to answer the question.
'''

BASELINE_PROMPT='''
**Note**:
1. Please answer within 10 words or fewer. If the answer is Yes or No, just say Yes or No
2. Do not include any special characters like <answer> or <answer/>."

With provided related documents:
<documents_section>
{docs_section}
</documents_section>

Answer the question:
<question_section>
{question}
</question_section>
'''

def prompt_generator(
    chunk_id_text_dict: Dict[Any, str],
    reordered_inputs: List[Dict[str, Any]],
    tokenizer=None,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    apply_template: bool = False
) -> Tuple[List[str], List[Any], List[Any]]:
    """
    Generate prompts from reordered inputs with document context.
    
    Args:
        chunk_id_text_dict: Mapping from document ID to document text
        reordered_inputs: List of reordered input dictionaries
        tokenizer: Pre-loaded tokenizer instance for chat template (optional)
        model_name: Model name to load tokenizer from (optional)
        system_prompt: Optional system prompt for chat template
        apply_template: Whether to apply chat template (default False)
    
    Returns:
        Tuple of (prompts, qids, answers)
    """
    def format_docs(reordered_doc_ids):
        """Format documents section using doc IDs"""
        docs_section = ""
        for doc_id in reordered_doc_ids:
            # Try both the original doc_id and string version to support both int and string IDs
            if doc_id in chunk_id_text_dict:
                content = chunk_id_text_dict[doc_id]
                docs_section += f"[Doc_{doc_id}] {content}\n\n"
            elif str(doc_id) in chunk_id_text_dict:
                content = chunk_id_text_dict[str(doc_id)]
                docs_section += f"[Doc_{doc_id}] {content}\n\n"
            else:
                print(f"Warning: Doc_{doc_id} not found")
        return docs_section.strip()
        
    def format_importance(original_doc_order):
        """Format importance ranking"""
        return " > ".join([f"[Doc_{doc_id}]" for doc_id in original_doc_order])
    
    prompts = []
    qids = [i["qid"] for i in reordered_inputs]
    answers = [i["answer"] for i in reordered_inputs]
    

    for reordered_input in reordered_inputs:
        reordered_doc_ids = reordered_input['top_k_doc_id']
        original_doc_order = reordered_input['orig_top_k_doc_id']
        question = reordered_input['question']

        docs_section = format_docs(reordered_doc_ids)
        importance_ranking = format_importance(original_doc_order)

        prompt = PROMPT_TEMPLATE.format(
            docs_section=docs_section,
            question=question,
            importance_ranking=importance_ranking
        )
        
        # Apply chat template if requested
        if apply_template:
            prompt = apply_chat_template(
                prompt,
                tokenizer=tokenizer,
                model_name=model_name,
                system_prompt=system_prompt
            )

        prompts.append(prompt)
        
    return prompts, qids, answers

def prompt_generator_baseline(
    chunk_id_text_dict: Dict[Any, str],
    inputs: List[Dict[str, Any]],
    tokenizer=None,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    apply_template: bool = False
) -> Tuple[List[str], List[Any], List[Any]]:
    """
    Generate baseline prompts from inputs with document context (no reordering).
    
    Args:
        chunk_id_text_dict: Mapping from document ID to document text
        inputs: List of input dictionaries
        tokenizer: Pre-loaded tokenizer instance for chat template (optional)
        model_name: Model name to load tokenizer from (optional)
        system_prompt: Optional system prompt for chat template
        apply_template: Whether to apply chat template (default False)
    
    Returns:
        Tuple of (prompts, qids, answers)
    """
    def format_docs(doc_ids):
        """Format documents section using doc IDs"""
        docs_section = ""
        for doc_id in doc_ids:
            if doc_id in chunk_id_text_dict:
                content = chunk_id_text_dict[doc_id]
                docs_section += f"[Doc_{doc_id}] {content}\n\n"
            else:
                print(f"Warning: Doc_{doc_id} not found")
        return docs_section.strip()
    
    prompts = []
    qids = [i["qid"] for i in inputs]
    answers = [i["answer"] for i in inputs]
    

    for _input in inputs:
        original_doc_order = _input['top_k_doc_id']
        question = _input['text']

        docs_section = format_docs(original_doc_order)
        # importance_ranking = format_importance(original_doc_order)

        prompt = BASELINE_PROMPT.format(
            docs_section=docs_section,
            question=question
        )
        
        # Apply chat template if requested
        if apply_template:
            prompt = apply_chat_template(
                prompt,
                tokenizer=tokenizer,
                model_name=model_name,
                system_prompt=system_prompt
            )

        prompts.append(prompt)
        
    return prompts, qids, answers