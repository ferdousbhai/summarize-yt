from pathlib import Path

import fastapi
import fastapi.staticfiles
from modal import App, Function, Mount, asgi_app

app = App("fastapi-react-app")

web_app = fastapi.FastAPI()
