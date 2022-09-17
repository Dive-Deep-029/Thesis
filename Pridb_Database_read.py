"""
Read pridb
# https://github.com/vallen-systems/pyVallenAE/blob/master/src/vallenae/io/schema_templates/pridb.sql
==========
"""
import os

import matplotlib.pyplot as plt

import vallenae as vae

from tabulate import tabulate


# Will take the path of the database from local path
# HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
HERE = "D:\Thesis\Python"
PRIDB = os.path.join(HERE, "sample.pridb")


# Open pridb database for analysing data in primary database(pridb)
# ----------
pridb = vae.io.PriDatabase(PRIDB)   # class vallenae.io.PriDatabase(filename, mode='ro'), pridb is object

print("Tables in database: ", pridb.tables())
print("Number of rows in data table (ae_data): ", pridb.rows())
print("Set of columns ", pridb.columns())
print("Set of all channels: ", pridb.channel())
print("info", pridb.globalinfo())
print("filedinfo", pridb.fieldinfo())



# Read hits to Pandas DataFrame
# -----------------------------
df_hits = pridb.read_hits()
# Print a few columns
# https://pyvallenae.readthedocs.io/en/latest/generated/vallenae.io.HitRecord.html
# https://pyvallenae.readthedocs.io/en/latest/_modules/vallenae/io/datatypes.html#HitRecord
# print(df_hits[["time", "channel", "amplitude", "counts", "energy"]])
print(tabulate(df_hits, headers = 'keys', tablefmt = 'psql'))

df = pridb.read()
print(tabulate(df, headers = 'keys', tablefmt = 'psql'))



# Query Pandas DataFrame
# ----------------------
# DataFrames offer powerful features to query and aggregate data,
# e.g. plot summed energy per channel
ax = df_hits.groupby("channel").sum()["energy"].plot.bar(figsize=(8, 3)) # figsize() takes two parameters- width and height (in inches)
ax.set_xlabel("Channel")
ax.set_ylabel("Summed Energy [eu = 1e-14 V²s]")
plt.tight_layout()  # automatically adjusts subplot params so that the subplot(s) fits in to the figure area
plt.show()


# Read markers
# ------------
df_markers = pridb.read_markers()
# print("markers", df_markers)
print(tabulate(df_markers, headers = 'keys', tablefmt = 'psql'))


# Read parametric data
# --------------------
df_parametric = pridb.read_parametric()
# print("param", df_parametric)
print(tabulate(df_parametric, headers = 'keys', tablefmt = 'psql'))

# df_hitrecord = vae.io.HitRecord()
# print("hits", df_hitrecord)

