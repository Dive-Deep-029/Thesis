"""
Read pridb
# https://github.com/vallen-systems/pyVallenAE/blob/master/src/vallenae/io/schema_templates/pridb.sql
==========
"""
import os
import vallenae as vae
from tabulate import tabulate


# Will take the path of the database from local path
HERE = "D:\Thesis\\acoustics\Data collected\\08-02-23"
PRIDB = os.path.join(HERE, "NBK7_700_0_7_30min.pridb")

# Open pridb database for analysing data in primary database(pridb)
# ----------
pridb = vae.io.PriDatabase(PRIDB)   # class vallenae.io.PriDatabase(filename, mode='ro'), pridb is object


print("info", pridb.globalinfo())
print("filedinfo", pridb.fieldinfo())


# Read hits to Pandas DataFrame
# -----------------------------
#features defined in vallen system - https://pyvallenae.readthedocs.io/en/stable/_modules/vallenae/features/acoustic_emission.html#rise_time

df_hits = pridb.read_hits()

# Print a few columns
# https://pyvallenae.readthedocs.io/en/latest/generated/vallenae.io.HitRecord.html
# https://pyvallenae.readthedocs.io/en/latest/_modules/vallenae/io/datatypes.html#HitRecord
print(df_hits[["time", "channel", "amplitude", "counts", "energy"]])

print(tabulate(df_hits, headers = 'keys', tablefmt = 'psql'))

