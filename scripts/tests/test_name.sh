find ./tests -name "*.py" -type f | while read file
do
  filename=$(basename "$file")
  dir=$(dirname "$file")
  if [[ $filename != test_* ]]; then
    mv "$file" "$dir/test_$filename"
    printf "\e[1;34mRenamed: \e[0m$file \e[1;32mto\e[0m $dir/test_$filename\n"
  fi
done