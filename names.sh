grep -h  -P -o "([a-zA-Z]+)" -r * |sort |uniq -c |sort -n >names.txt
