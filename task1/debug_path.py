from __future__ import annotations

import logging, os, pathlib, sys

def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> int:
    configure_logging()
    logger = logging.getLogger("debug_path")

    logger.info("Current Working Directory: %s", os.getcwd())
    logger.info("Sys Path:")
    for p in sys.path:
        logger.info("  %s", p)

    logger.info("File resolution:")
    try:
        from kontakt_qc import types  # type: ignore
        logger.info("kontakt_qc.types file: %s", types.__file__)
    except ImportError:
        logger.exception("ImportError while importing kontakt_qc.types")

    logger.info("Directory Listing of github_repos:")
    try:
        path = pathlib.Path(os.getcwd())
        while path.name != "github_repos" and path.parent != path:
            path = path.parent

        if path.name == "github_repos":
            logger.info("Found github_repos at: %s", path)
            logger.info("Contents:")
            for item in path.iterdir():
                logger.info("  %s", item.name)
        else:
            logger.warning("Could not find github_repos in ancestry of %s", os.getcwd())
    except Exception:
        logger.exception("Error checking directories")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
