# Development server commands

## Start jupyter container for data science activities (run ipynb
```bash
docker compose up jupyter
```
## Start jupyter notebook and sync with the local directory (for active development)
** Note! Watch mode does not seem to work on WSL.
```bash
docker compose watch jupyter
```

# Ray training results
The pickle file is stored in the mounted volume at:
```bash
/var/lib/docker/volumes/dev_shared/_data
```
