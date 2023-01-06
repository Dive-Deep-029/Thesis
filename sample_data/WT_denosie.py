# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 22:04:04 2022

@author: hemli
"""
from pywt import wavedec

a = [1, 2, 3, 4, 5, 6, 7, 8]

coeffs = wavedec(a, 'db1', level=2)
