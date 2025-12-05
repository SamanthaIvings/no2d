library(tidyverse)
library(MASS)
library(plyr)
library(dplyr)
library(data.table)
library(reshape2)

font_import()
loadfonts(device = "win")

##### PARAMETER DEFINITIONS #####

# plot aesthetics
font <- "Times New Roman"

# working directory
setwd("") # INSERT WORKING DIRECTORY

# files to be read/written in this script
args <- c("inputs/demand.csv",             #1   input
          "inputs/demand_weekends.csv",    #2   input
          "inputs/OD_matrix.csv",          #3   input
          "inputs/OD_matrix_weekends.csv") #4   input

demand_wd <- read_csv(args[1])
demand_we <- read_csv(args[2])
OD_wd <- read_csv(args[3])
OD_we <- read_csv(args[4])

sum(demand_wd)
sum(demand_we)
sum(demand_we)*100/sum(demand_wd)
(sum(demand_wd)-sum(demand_we))*100/sum(demand_wd)
nrow(demand_wd)
nrow(demand_we)
nrow(demand_we)*100/nrow(demand_wd)
(nrow(demand_wd)-nrow(demand_we))*100/nrow(demand_wd)
nrow(demand_wd)-nrow(demand_we)
(nrow(demand_wd)-nrow(demand_we))*100/(345*345)

OD_comparison <- OD_wd-OD_we

