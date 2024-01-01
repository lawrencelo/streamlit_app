def extract_text_segment(reference_text, target_string, max_word_count):
    # Find the index of the target string
    start_index = reference_text.find(target_string)
    if start_index == -1:
        return "Target string not found in the document."

    # Calculate the end index of the target string
    end_index = start_index + len(target_string)

    # Initialize counters and pointers
    word_count = 0
    left_index = start_index
    right_index = end_index

    # Expand left and right while counting words, and stop if limit is reached
    while word_count < max_word_count:
        # Expand to the left
        if left_index > 0:
            left_index = reference_text.rfind(' ', 0, left_index)
            if left_index == -1:  # Start of the document
                left_index = 0
                break
            # Check if it's the start of a sentence
            if reference_text[left_index] in '.!?':
                word_count += len(reference_text[left_index:start_index].split())

        # Expand to the right
        if right_index < len(reference_text):
            right_index = reference_text.find(' ', right_index)
            if right_index == -1:  # End of the document
                right_index = len(reference_text)
                break
            # Check if it's the end of a sentence
            if reference_text[right_index] in '.!?':
                word_count += len(reference_text[end_index:right_index].split())

        # Update start and end indices for the target string
        start_index = left_index
        end_index = right_index

        # Break if both ends have reached the document boundaries
        if left_index == 0 and right_index == len(reference_text):
            break

    return reference_text[left_index:right_index].strip()

# Example usage
reference_text = "Your reference text document goes here..."
target_string = "string 1"
max_word_count = 3000

extracted_segment = extract_text_segment(reference_text, target_string, max_word_count)
print(extracted_segment)
