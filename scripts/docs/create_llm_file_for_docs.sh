#!/bin/bash

# Set up logging
LOG_FILE="docs_compilation.log"
OUTPUT_FILE="combined_docs.txt"

# Initialize log file
echo "$(date): Starting documentation compilation" > "$LOG_FILE"

# Create/clear output file
> "$OUTPUT_FILE"

# Function to determine file type and handle accordingly
process_file() {
    local file="$1"
    
    # Get file extension
    extension="${file##*.}"
    
    echo "$(date): Processing $file" >> "$LOG_FILE"
    
    case "$extension" in
        md|markdown)
            echo "# $(basename "$file")" >> "$OUTPUT_FILE"
            cat "$file" >> "$OUTPUT_FILE"
            echo -e "\n\n" >> "$OUTPUT_FILE"
            ;;
        txt)
            echo "# $(basename "$file")" >> "$OUTPUT_FILE"
            cat "$file" >> "$OUTPUT_FILE"
            echo -e "\n\n" >> "$OUTPUT_FILE"
            ;;
        *)
            echo "$(date): Skipping $file - unsupported format" >> "$LOG_FILE"
            return
            ;;
    esac
    
    echo "$(date): Successfully processed $file" >> "$LOG_FILE"
}

# Find and process all documentation files
find ../docs -type f \( -name "*.md" -o -name "*.txt" -o -name "*.markdown" \) | while read -r file; do
    process_file "$file"
done

# Log completion
echo "$(date): Documentation compilation complete" >> "$LOG_FILE"
echo "$(date): Output saved to $OUTPUT_FILE" >> "$LOG_FILE"

# Print summary
echo "Documentation compilation complete. Check $LOG_FILE for details."