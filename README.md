# sana_server

Unified web server for all Sana models. Lives at `/Desktop/sana/sana_server/`.

## Directory layout

```
/Desktop/sana/
  sana_server/          ← this folder
    server.py           ← run this
    static/
      index.html        ← the web UI
  sana_XXM/        ← untouched model folder
```

The server scans `../` for `sana_*` folders that contain a checkpoint.
No changes needed to existing model folders.

## Running

```bash
cd /Desktop/sana/sana_server
source ../sana_23M/venv/bin/activate   # or any venv with torch + sentencepiece

# Start (no model auto-loaded — pick one in the UI)
python server.py

# Start with a model pre-loaded
python server.py --load sana_23M_v2

# Custom port
python server.py --port 8100 --load sana_23M_v2
```

## systemd service

Update `/etc/systemd/system/sana-chat.service`:

```ini
[Unit]
Description=Sana Chat Server
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/home/user/Desktop/sana/sana_server
ExecStart=/home/user/Desktop/sana/sana_23M_v1/venv/bin/python server.py --port 8100 --load sana_23M_v2
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

## API

| Method | Path          | Description                                       |
| ------ | ------------- | ------------------------------------------------- |
| GET    | `/api/models` | List available models + active                    |
| POST   | `/api/load`   | `{"model_id": "sana_23M_v2"}` — load a model      |
| POST   | `/api/chat`   | `{"message": "...", "history": [...]}` — generate |
| POST   | `/api/reset`  | Clear server-side state                           |
| GET    | `/api/status` | Current model info + loading state                |
| GET    | `/`           | Web UI                                            |

## Adding new models

Just create the folder with the standard structure under `/Desktop/sana/`.
The server discovers it automatically on next `/api/models` call.
Add metadata to `MODEL_META` in `server.py` for a nice label and description.
