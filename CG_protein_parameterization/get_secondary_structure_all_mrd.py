#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yang Jiang @ PSU
"""

'''
Michael DeLyser:
    This was originally opt_nscal.py
    I needed to generate the "secondary_struc_defs.txt" file because it is used for calc_native_contact_fraction.pl
    If there was a standalone script that simply output this file somewhere, I missed it.
    Anyways, this is basically a gutted version of opt_nscal.py that does the minimum stuff up until it calls get_secondary_structure() and then exits
'''

import os, time, traceback, io, sys, getopt, multiprocessing, random
import parmed as pmd
import numpy as np

usage = '''
  Usage: python get_secondary_structure.py 
                --input | -i <input.pdb> for secondary structure assignment
                --output | -o <secondary_struc_defs.txt> alternative output name
                --savepdb | -s save the clean pdb file (default is to delete it)
'''

################# Arguments #################
if len(sys.argv) == 1:
    print(usage)
    sys.exit()

input_pdb = ""
output_filename = "secondary_struc_defs.txt"
savepdb = False
try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:s", ["help", "input=","output=","savepdb"])
except getopt.GetoptError:
    print(usage)
    sys.exit()
for opt, arg in opts:
    if opt in ('-h', "--help"):
        print(usage)
        sys.exit()
    elif opt in ("-i", "--input"):
        input_pdb = arg
    elif opt in ("-o", "--output"):
        output_filename = arg
    elif opt in ("-s", "--savepdb"):
        savepdb = True

has_error = False
if input_pdb == "":
    print('Error: You must specify the path to the all-atom pdb file for the secondary structure assignment.')
    has_error = True
    
if has_error:
    print(usage)
    sys.exit()


################### Functions #########################
def clean_pdb(input_pdb):
    AA_name_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
                    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
                    'HIE', 'HID', 'HIP'];
    #print("-> Cleaning PDB file %s"%input_pdb)
    name = input_pdb.split('/')[-1].split('.pdb')[0]
    struct = pmd.load_file(input_pdb)
    sel_idx = np.zeros(len(struct.atoms))
    for idx, res in enumerate(struct.residues):
        res.number = idx+1
        if res.name in AA_name_list:
            for atm in res.atoms:
                sel_idx[atm.idx] = 1
    struct[sel_idx].save(name+'_clean.pdb', overwrite=True, altlocs="occupancy")
    #print("   PDB file cleaned")
    return name+'_clean.pdb'
    
def get_secondary_structure(pdb,ofnm):
    #print("-> Getting secondary structure information")
    
    screen_out = os.popen('stride '+pdb).readlines()
    # MRD catch rare case of stride outputting a Cycle Anti line
    CycleAntiLines = []
    while screen_out[0].startswith('Cycle Anti'):
        CycleAntiLines.append(screen_out[0])
        del(screen_out[0])

    if not screen_out[0].startswith('REM  -------------------- Secondary structure summary -------------------  ~~~~'):
        print(''.join(screen_out))
        sys.exit()

    pdb_struct = pmd.load_file(pdb)
    sec_ele_list = []
    for line in screen_out:
        line = line.strip()
        if line.startswith('LOC '):
            sec_name = line[5:17].strip()
            if 'Helix' in sec_name or 'Strand' in sec_name:
                chainid = line[28]
                start_resnum = int(line[21:27].strip())
                end_resnum = int(line[39:45].strip())
                length = end_resnum - start_resnum + 1
                if length >= 2: # MRD ONLY CHANGE
                    start_resid = np.nan
                    end_resid = np.nan
                    for res in pdb_struct.residues:
                        if res.chain == chainid and res.number == start_resnum:
                            start_resid = res.idx
                        elif res.chain == chainid and res.number == end_resnum:
                            end_resid = res.idx
                        if not np.isnan(start_resid) and not np.isnan(end_resid):
                            break
                    if np.isnan(start_resid) or np.isnan(end_resid):
                        print('Error: Cannot find residue %d or %d in chain %s in %s.'%(start_resnum, end_resnum, chainid, pdb))
                        sys.exit()
                    sec_ele_list.append([start_resid+1, end_resid+1])
    sec_ele_list.sort(key=lambda x: x[0])
    
    f = open(ofnm, 'w')
    for idx, sec_ele in enumerate(sec_ele_list):
        f.write('%d %d %d\n'%(idx+1, sec_ele[0], sec_ele[1]))
    f.close()
    if (len(CycleAntiLines) > 0):
        print("Note: STRIDE output the following lines before the secondary structure:")
        print(''.join(CycleAntiLines))
    #print("   Done.")


########################### MAIN #########################################

clean_pdb = clean_pdb(input_pdb)
get_secondary_structure(clean_pdb,output_filename)
if not savepdb:
    os.remove(clean_pdb)
