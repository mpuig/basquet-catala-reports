import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s | %(name)s | %(asctime)s | %(message)s",
    stream=sys.stderr,
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger("match_report_generator")
