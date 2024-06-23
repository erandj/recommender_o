import builtins
import functools
import logging

from fastapi.responses import RedirectResponse
from src.api import routers
from fastapi import FastAPI
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


test_data_folder = "test_data/"

origins = [
    "https://animila.ru",
    "https://www.animila.ru",
    "https://api.animila.ru",
]

app = FastAPI()
app.include_router(routers.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

model_files_directory = Path("/app/src/user_models")
app.mount("/files/models", StaticFiles(directory=model_files_directory), name="model_files")

@app.get("/", include_in_schema=False)
async def root():
    response = RedirectResponse(url='/docs#/')
    return response


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG, 
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("py_log.log"),
            logging.StreamHandler()
        ])
    
    logging.info("Logger is running")


def log_print(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info("print function called:")
        return func(*args, **kwargs)
    return wrapper


def main():
    setup_logging()
    builtins.print = log_print(builtins.print)


if __name__ == "__main__":
    main()
    
