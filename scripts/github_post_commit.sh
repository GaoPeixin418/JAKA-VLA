#!/bin/sh
set -eu

project_root=$(git rev-parse --show-toplevel)
if [ "$project_root" != "/data/jaka-vla" ]; then
    printf 'Refusing GitHub autosync outside /data/jaka-vla: %s\n' "$project_root" >&2
    exit 2
fi

git_dir=$(git rev-parse --absolute-git-dir)
export GIT_SSH_COMMAND="ssh -i $git_dir/jaka-vla-deploy-key -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$git_dir/github_known_hosts"

if git -C "$project_root" push origin HEAD:refs/heads/main; then
    printf 'JAKA-VLA commit pushed to GitHub main.\n'
else
    status=$?
    printf 'JAKA-VLA commit remains local; automatic push failed (exit %s).\n' "$status" >&2
    exit "$status"
fi
