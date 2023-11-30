find ./tests -name "*.py" -type f | while read file
do
  filename=$(basename "$file")
  dir=$(dirname "$file")
  if [[ $filename != test_* ]]; then
    mv "$file" "$dir/test_$filename"
  fi
done