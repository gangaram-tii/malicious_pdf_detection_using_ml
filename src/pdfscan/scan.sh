#!/run/current-system/sw/bin/bash

# Set the file name to be processed (can be passed as an argument)
FILE_NAME=$1
SERVER=$2

# Ensure the file name is provided
if [ -z "$FILE_NAME" ]; then
  echo "Please provide a file name as an argument."
  exit 1
fi

# Send the file name to the Python service via the socket connection
RESPONSE=$(echo -n "$FILE_NAME" | nc $SERVER 65432)

# Check the response and print result
echo "Response from server: $RESPONSE"

