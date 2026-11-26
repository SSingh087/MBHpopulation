"""Put astrophysical_setup on sys.path: its modules import each other by bare name (``from config import ...``)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "MBH_population_from_EMRI_TDE", "astrophysical_setup"))
