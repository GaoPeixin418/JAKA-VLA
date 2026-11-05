#!/bin/sh
set -eu

expected_root=/data/jaka-vla
project_root=$(cd "$(dirname "$0")/.." && pwd -P)
if [ "$project_root" != "$expected_root" ]; then
    printf 'Refusing autosync install outside %s: %s\n' "$expected_root" "$project_root" >&2
    exit 2
fi

git_dir=$(git rev-parse --absolute-git-dir)
private_key="$git_dir/jaka-vla-deploy-key"
known_hosts="$git_dir/github_known_hosts"
hook="$git_dir/hooks/post-commit"
test -s "$private_key" || { printf 'Missing project Deploy Key: %s\n' "$private_key" >&2; exit 1; }
test -s "$known_hosts" || { printf 'Missing pinned GitHub host key file: %s\n' "$known_hosts" >&2; exit 1; }
chmod 600 "$private_key"

git config --local remote.origin.url git@github.com:GaoPeixin418/JAKA-VLA.git
git config --local core.sshCommand "ssh -i $private_key -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$known_hosts"
export GIT_SSH_COMMAND="ssh -i $private_key -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$known_hosts"
git ls-remote origin refs/heads/main >/dev/null
git fetch origin main:refs/remotes/origin/main
git merge-base --is-ancestor refs/remotes/origin/main HEAD || {
    printf 'Local JAKA-VLA history is not a fast-forward from GitHub main.\n' >&2
    exit 1
}

install -m 755 scripts/github_post_commit.sh "$hook"
printf 'Installed JAKA-VLA-only post-commit sync hook.\n'
