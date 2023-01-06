# -*- coding: utf-8 -*-
"""
File: DemoConvertTradbToTxt.py
Demo for converting waveforms in tradb-file to Textfiles with Python
Author: Vallen System GmbH
Date: 21. Feb 2020
www.vallen.de, info@vallen.de
Tested with Python3.5 

"""

import array
import numpy
import matplotlib.pyplot as grafix
import sqlite3

strZeilenumbruch='\r\n' # for Windows

tradb_file= "Demo6Waveforms"  # tradb filename without extension
db = sqlite3.connect(str(tradb_file) + ".tradb")
cursor= db.cursor()           # init cursor in database
sql= "Select Data, TR_mV, SampleRate, TRAI, Chan FROM view_tr_data"
# for selecting single TRAI:
#sql= "Select Data, TR_mV, SampleRate, TRAI FROM view_tr_data WHERE TRAI = 4"

try:
    cursor.execute(sql)
    row= cursor.fetchone() #read one dataset
    while row is not None:
        trai= row[3]
        chan= row[4]
        print ("TRAI= ", trai, "Chan= ", chan)
        vecSamples = array.array('h',bytes(row[0]))
        vecTR = numpy.multiply(vecSamples,row[1]) # converts samples in mV
        vecT = numpy.arange(len(vecTR ))          # create a vector for time axis
        vecT = numpy.multiply(vecT,1E6/row[2])      # convert to timescale in microsec
        
        # Graphic representation for checking, remove comment if needed
        grafix.plot(vecT,vecTR)                   # create plot
        grafix.show()                             # display plot    
        
        ofname=tradb_file + "_TRAI_" + str(trai) + "_CHAN_" + str(chan) + ".txt"
        ofile = open(ofname, "w")
        strHeader= "Datafile: " + tradb_file + ", Trai: " + str(trai) + ", Chan:" + str(chan) + strZeilenumbruch
        strHeader+= "Time (microsec)" + '\t' + "Amplitude (mV)"
        # hint to format fmt, use e instead for f for scientific format
        numpy.savetxt(ofname, numpy.c_[vecT, vecTR], fmt='%.3f\t%.3f', delimiter='\t', newline= strZeilenumbruch, header= strHeader, comments='' )
        ofile.close
        row= cursor.fetchone() # read next dataset, returns None if end
except sqlite3.Error as e:
    print ("Error sqlite:", e.args[0])

cursor.close()    
db.close()

