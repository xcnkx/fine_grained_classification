#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 02:19:33 2018

@author: n-kamiya
"""

import sqlite3

image_txt = "/home/n-kamiya/datasets/CUB2002011/CUB_200_2011"
images_path = "/home/n-kamiya/datasets/CUB2002011/CUB_200_2011/images"
root = image_txt

def create_cub_db():

    
    dbpath = "/home/n-kamiya/fine-grained_project/database/fg_project.db"
    
    con = sqlite3.connect(dbpath)
    
    cr = con.cursor()
    
    
    cr.execute("DROP TABLE IF EXISTS CUB2002011")
    cr.execute("""CREATE TABLE CUB2002011
               (id INTEGER PRIMARY KEY,
               class INTEGER,
               train INTEGER,
               path TEXT);""")
    
    
    with open("%s/images.txt" % root) as f, open("%s/image_class_labels.txt" % root) as g, open("%s/train_test_split.txt" % root) as h:
        for line in f.readlines():
            data = line.split()
            cr.execute("INSERT INTO CUB2002011 VALUES (?,NULL,NULL,?)",(data[0], data[1]))
        for line in g.readlines():
            data = line.split()
            cr.execute("UPDATE CUB2002011 SET class=? WHERE id=?", (data[1], data[0]))
        for line in h.readlines():
            data = line.split()
            cr.execute("UPDATE CUB2002011 SET train=? WHERE id=?", (data[1],data[0]))
            
    return con , cr