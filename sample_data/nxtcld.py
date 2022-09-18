# pyncclient

import nextcloud_client
import os

nc = nextcloud_client.Client('https://nextcloud.th-deg.de')

nc.login('hs17334', 'h814334-s1996')

print(os.getcwd())
#nc.mkdir('testdir')
#nc.put_file('testdir/sample.pridb', 'D:\\Thesis\\Python\\sample.pridb')

# nc.get_file("testdir/RDBI.docx")
print(nc.get_file_contents("testdir"))

