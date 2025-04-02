#!/bin/bash
                                                                              # Prompt the user for the image search query
read -p "Enter your image search query: " query

# URL encode the query to handle spaces and special characters properly
encoded_query=$(printf '%s' "$query" | sed 's/ /+/g; s/[^a-zA-Z0-9_+.-]/\\&/g')

# Construct the Google Images URL with the encoded query
url="https://www.google.com/images?q=$encoded_query"

# Open the URL using xdg-open
xdg-open "$url"

echo "Opening Google Images for: '$query'"
