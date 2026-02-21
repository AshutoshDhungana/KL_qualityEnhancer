#!/usr/bin/env python3
"""
Example script demonstrating the enhanced question merging pipeline.
"""
import sys
import json
from merge_utils import merge_qa_pairs, merge_qa_pairs_by_ids, get_qa_pair_by_id

def print_qa_pair(pair, title="QA Pair"):
    """Print a QA pair in a readable format"""
    print(f"\n=== {title} ===")
    print(f"ID: {pair.get('id', 'N/A')}")
    print(f"Question: {pair.get('question', '')}")
    print(f"Answer: {pair.get('answer', '')}")
    if 'sources' in pair:
        print(f"Sources: {', '.join(pair['sources'])}")
    print("="*40)

def merge_by_objects():
    """Demonstrate merging using QA pair objects"""
    print("\n[1] Merging QA pairs by objects")

    # Example QA pairs
    qa1 = {
        "id": "q1",
        "question": "How do I reset my password?",
        "answer": "To reset your password, click on the 'Forgot Password' link on the login page and follow the instructions."
    }
    
    qa2 = {
        "id": "q2",
        "question": "What should I do if I forgot my password?",
        "answer": "If you forgot your password, use the password recovery option. You'll receive an email with instructions."
    }
    
    # Print original QA pairs
    print_qa_pair(qa1, "Original QA Pair 1")
    print_qa_pair(qa2, "Original QA Pair 2")
    
    # Merge without priority
    merged_no_priority = merge_qa_pairs(qa1, qa2)
    print_qa_pair(merged_no_priority, "Merged QA Pair (No Priority)")
    
    # Merge with priority on first pair
    merged_priority1 = merge_qa_pairs(qa1, qa2, priority="pair1")
    print_qa_pair(merged_priority1, "Merged QA Pair (Priority on Pair 1)")
    
    # Merge with priority on second pair
    merged_priority2 = merge_qa_pairs(qa1, qa2, priority="pair2")
    print_qa_pair(merged_priority2, "Merged QA Pair (Priority on Pair 2)")

def merge_by_ids():
    """Demonstrate merging using QA pair IDs"""
    print("\n[2] Merging QA pairs by IDs")
    
    # Example IDs (these should be valid IDs in your database)
    # For demonstration, we'll use the example IDs
    cq_ids = ["q1", "q2", "q3"]
    
    print(f"Looking for QA pairs with IDs: {', '.join(cq_ids)}")
    
    # Try to retrieve each QA pair
    for cq_id in cq_ids:
        qa_pair = get_qa_pair_by_id(cq_id)
        if qa_pair:
            print_qa_pair(qa_pair, f"Retrieved QA Pair {cq_id}")
        else:
            print(f"\nQA pair with ID {cq_id} not found")
    
    # For demonstration, since get_qa_pair_by_id may not find real pairs,
    # we'll use mock QA pairs for the rest of the example
    mock_qa_pairs = {
        "q1": {
            "id": "q1",
            "question": "How do I reset my password?",
            "answer": "To reset your password, click on the 'Forgot Password' link on the login page."
        },
        "q2": {
            "id": "q2",
            "question": "What should I do if I forgot my password?",
            "answer": "If you forgot your password, use the password recovery option."
        },
        "q3": {
            "id": "q3",
            "question": "Can I recover my account if I forget my password?",
            "answer": "Yes, account recovery is possible through the password reset procedure."
        }
    }
    
    # Simulate the merge_qa_pairs_by_ids function with our mock data
    print("\nSimulating merger with mock data:")
    
    # Without priority
    print("\nMerging all QA pairs without priority:")
    all_pairs = [mock_qa_pairs[cq_id] for cq_id in cq_ids if cq_id in mock_qa_pairs]
    # This would be done by merge_qa_pairs_by_ids in a real scenario
    from merge_utils import QuestionMerger
    merger = QuestionMerger()
    merged_all = merger.merge_qa_pair(all_pairs)
    merged_all["id"] = f"merged_{'_'.join(cq_ids)}"
    print_qa_pair(merged_all, "Merged QA Pair (No Priority)")
    
    # With priority on second question
    print("\nMerging all QA pairs with priority on 'q2':")
    priority_id = "q2"
    priority_idx = cq_ids.index(priority_id) if priority_id in cq_ids else None
    merged_priority = merger.merge_qa_pair(all_pairs, priority_idx)
    merged_priority["id"] = f"merged_{'_'.join(cq_ids)}_priority_{priority_id}"
    print_qa_pair(merged_priority, f"Merged QA Pair (Priority on {priority_id})")

def main():
    """Main function to run examples"""
    print("Question Merging Pipeline Example")
    print("================================")
    
    try:
        # Run the object-based merging example
        merge_by_objects()
        
        # Run the ID-based merging example
        merge_by_ids()
        
        print("\nExample completed successfully!")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 