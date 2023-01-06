"""
Read pridb
# https://github.com/vallen-systems/pyVallenAE/blob/master/src/vallenae/io/schema_templates/pridb.sql
==========
"""
import os
import pandas as df
import matplotlib.pyplot as plt

import vallenae as vae

from tabulate import tabulate


# Will take the path of the database from local path
# HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
HERE = "D:\Thesis\\acoustics\Data collected\\23-11-2022\water"
PRIDB = os.path.join(HERE, "sensor_5cm_from_lens_CM.pridb")

#HERE = "D:\Thesis\\acoustics\Data collected\9-11-2022\continous_mode_with_lens"
#PRIDB = os.path.join(HERE, "one_scratch_CM.pridb")


# Open pridb database for analysing data in primary database(pridb)
# ----------
pridb = vae.io.PriDatabase(PRIDB)   # class vallenae.io.PriDatabase(filename, mode='ro'), pridb is object

""""
print("Tables in database: ", pridb.tables())
print("Number of rows in data table (ae_data): ", pridb.rows())
print("Set of columns ", pridb.columns())
print("Set of all channels: ", pridb.channel())
"""""
print("info", pridb.globalinfo())
print("filedinfo", pridb.fieldinfo())




# Read hits to Pandas DataFrame
# -----------------------------
#features defined in vallen system - https://pyvallenae.readthedocs.io/en/stable/_modules/vallenae/features/acoustic_emission.html#rise_time

#df_hits = pridb.read_hits(channel=1, time_start=3.37, time_stop=3.44,set_id=None, query_filter=None)
df_hits = pridb.read_hits()
def csv():
    df_hits.to_csv("D:\Thesis\\acoustics\Data collected\9-11-2022\continous mode_wo_lens_only_sensor\CM_senor_only_$_env.csv"
          , index = False, encoding='utf-8')
#csv()
# Print a few columns
# https://pyvallenae.readthedocs.io/en/latest/generated/vallenae.io.HitRecord.html
# https://pyvallenae.readthedocs.io/en/latest/_modules/vallenae/io/datatypes.html#HitRecord
#print(df_hits[["time", "channel", "amplitude", "counts", "energy"]])
print(tabulate(df_hits, headers = 'keys', tablefmt = 'psql'))
print(df_hits["trai"])
trai = df_hits["trai"].tolist()
print("trai_list",trai)
print("trai_len",len(trai))


# Query Pandas DataFrame
# ----------------------
# DataFrames offer powerful features to query and aggregate data,
# e.g. plot summed energy per channel
"""
ax = df_hits.groupby("channel").sum()["energy"].plot.bar(figsize=(8, 3)) # figsize() takes two parameters- width and height (in inches)
ax.set_xlabel("Channel")
ax.set_ylabel("Summed Energy [eu = 1e-14 V²s]")
plt.tight_layout()  # automatically adjusts subplot params so that the subplot(s) fits in to the figure area
plt.show()
"""

# Read markers
# ------------
"""
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
"""
