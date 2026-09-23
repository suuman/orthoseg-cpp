import argparse
import os

import uvicorn

from app.core.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.config:
        os.environ["XRAY_CONFIG"] = os.path.abspath(args.config)
    config = load_config(args.config)
    uvicorn.run("app.main:app", host=config["server"]["host"], port=config["server"]["port"], workers=1)


if __name__ == "__main__":
    main()
