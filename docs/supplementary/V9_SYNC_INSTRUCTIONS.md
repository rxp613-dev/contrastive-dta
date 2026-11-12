# V9 GitHub / Zenodo Sync Instructions (Task 3.4.2)

Status on 2026-11-12: all files are staged and committed locally; the remote
push could NOT be executed because:

1. Direct HTTPS to github.com fails (TLS reset, curl rc=35);
2. SSH to github.com:22 and ssh.github.com:443 is closed;
3. Local proxies (127.0.0.1:7897 / :8080, HTTP and SOCKS5) establish CONNECT
   but the TLS exchange to GitHub fails (HTTP code 000);
4. Zenodo (zenodo.org) is blocked the same way;
5. The `gh` keyring token is reported invalid and no SSH key / PAT is
   available on this machine.

When network and credentials are restored, run:

```bash
cd /mnt/data/dta2/repo_for_github
git remote -v            # confirm origin = https://github.com/rxp613-dev/contrastive-dta.git
git status               # confirm the V9 commit is present
git push origin master   # remote default branch is master in this repo

# Then create a release so the GitHub-Zenodo integration archives it:
gh release create v1.1.0 --title "v1.1.0 - V9 convergent-protocol reassessment" \
  --notes "V9 reassessment package: Config C (max_epochs=1000) results,
  provenance tables, manuscript source, supplementary S3."
```

Zenodo will mint a version DOI for the new release; the concept DOI
10.5281/zenodo.23308152 remains unchanged. Update CITATION.cff with the new
version DOI after the release.
