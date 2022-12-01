#!/usr/bin/env python3

# Positions
FLOWER = 0
PLUME = 1
SANDS = 2
GOBLET = 3
CIRCLET = 4

# Properties
HP = "HP"
HPP = "HPP"
ATK = "ATK"
ATKP = "ATKP"
DEF = "DEF"
DEFP = "DEFP"
ER = "ER"
EM = "EM"
CR = "CR"
CDMG = "CDMG"
EDMG = "EDMG"
HB = "HB"

# Elements
# PHYSICAL
# PYRO
# HYDRO
# DENDRO
# ELECTRO
# ANEMO
# CYRO
# GEO

# https://nga.178.com/read.php?tid=25247146
ARTIFACT_MAIN_PROP_DIST = {
    FLOWER: {HP: 10000},
    PLUME: {ATK: 10000},
    SANDS: {HPP: 2668, ATKP: 2666, DEFP: 2666, ER: 1000, EM: 1000},
    GOBLET: {HPP: 1950, ATKP: 1950, DEFP: 1850, EDMG: 4000, EM: 250},
    CIRCLET: {HPP: 2200, ATKP: 2200, DEFP: 2200, CR: 1000, CDMG: 1000, HB: 1000, EM: 400},
}
ARTIFACT_MAIN_PROP_MAX = {
    HP: 4780.0,
    ATK: 311.2,
    HPP: 46.6,
    ATKP: 46.6,
    DEFP: 58.3,
    ER: 51.8,
    EM: 186.5,
    CR: 31.1,
    CDMG: 62.2,
    EDMG: 46.6,  # Physical is 58.3
    HB: 35.9,
}

ARTIFACT_ADDITIONAL_PROP_DIST = {HP: 1500, ATK: 1500, DEF: 1500, HPP: 1000, ATKP: 1000, DEFP: 1000, ER: 1000, EM: 1000, CR: 750, CDMG: 750}

# https://nga.178.com/read.php?tid=31774495
ADDITIONAL_PROP_BASE = {
    HP: 239.0,
    HPP: 4.66,
    ATK: 15.56,
    ATKP: 4.66,
    DEF: 18.52,
    DEFP: 5.83,
    ER: 5.18,
    EM: 18.65,
    CR: 3.11,
    CDMG: 6.22,
}
ADDITIONAL_PROP_LEVELS = [1.25, 1.125, 1.0, 0.875]

CHARACTER_VEC_MAPPING = ["lvl","base_hp","hp","base_atk","atk","base_def","def","er","em","edmg","cr","cdmg","hb"]
ARTIFACT_VEC_MAPPING = [HP,HPP,ATK,ATKP,DEF,DEFP,ER,EM,EDMG,CR,CDMG,HB]