#!/bin/bash

if [ -n "${BASH_SOURCE[0]}" ]; then
    script_dir=$(cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
else
    script_dir=$(dirname "$0")
fi

name=$(git config user.name)
email=$(git config user.email)

(
cat <<EOF
GIT_AUTHOR_NAME="$name"
GIT_AUTHOR_EMAIL=$email
GIT_COMMITTER_NAME="$name"
GIT_COMMITTER_EMAIL=$email
EOF
) > "$script_dir/.env"