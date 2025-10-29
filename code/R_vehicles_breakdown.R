library(tidyverse)
library(caret)
library(MASS)
library(plyr)
library(dplyr)
library(ggplot2)
library(data.table)
library(viridis)
library(lubridate)
library(reshape2)
library(glmnet)
library(minpack.lm)
library(RColorBrewer)
library(rmweather)
library(extrafont)

font_import()
loadfonts(device = "win")

##### PARAMETER DEFINITIONS #####

# plot aesthetics
font <- "Times New Roman"
bgdy <- "#940B13"

# how many iterations in the Frank-Wolfe algorithm
stepbreak <- 100000

# working directory
setwd("") # INSERT WORKING DIRECTORY

##### VEHICLES DATA #####

veh <- read_csv("inputs/vehiclesBreakDownFull.csv")

veh$ageRng[veh$ageRng=="15+"] <- ">15"
veh$ageRng <- factor(veh$ageRng, levels=c("<2","2-3","4-7","8-15",">15"))

veh$VehCat[veh$VehCat=="mBs/DSL"] <- "mBus/DSL"
veh$VehCat[veh$VehCat=="PCr/hePtr"] <- "PCr/hePTRL"

veh <- veh  %>%
  dplyr::mutate(TotalVehLog = log(TotalVeh))

vehSums <- veh %>%
  dplyr::group_by(VehCat) %>%
  dplyr::summarise(tot = sum(TotalVeh), tot_log = sum(TotalVehLog))
vehSums$tot <- round(vehSums$tot,1)
vehSums$tot_log <- round(vehSums$tot_log,1)

p <- ggplot() +
  geom_bar(data = veh, aes(x=VehCat, y=TotalVehLog, fill=ageRng),
           stat="identity", col="black") +
  geom_text(data = vehSums, aes(x=VehCat, y=tot_log+3, label=tot),
            size=3.25, col="black") +
  guides(col="none") +
  scale_fill_viridis(discrete=TRUE, direction=-1) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=17, family=font),
        panel.grid.major=element_line(color="grey"),
        axis.text.x=element_text(angle=90, vjust=0.5, hjust=1)) +
  expand_limits(x= c(0, 13)) +
  labs(x="Combined engine and fuel type", y="Log number of vehicles", fill="Age")
#p <- edit_colors(p, desaturate)
print(p)
mypath <- "plots/vehStacks_Log.png"
ggsave(mypath, plot = last_plot(), width = 8, height = 6, units = "in")

##### CAPACITY DATA #####

capPath <- 'inputs/TfL_capacities.csv'
cap <- read_csv(capPath)

cap$class_num <- ifelse(cap$class == "UAP1", 1,
                        ifelse(cap$class == "UAP2", 2,
                               ifelse(cap$class == "UAP3", 3,
                                      4)))
cap$road_class <- factor(cap$class_num)

p <- ggplot(cap, aes(x=width, y=capacity, col=class)) +
  geom_point() +
  geom_line(linewidth=0.75) +
  #scale_color_viridis(discrete=TRUE, direction=-1) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_continuous(breaks=seq(6,18,2), labels=as.character(seq(6,18,2))) +
  labs(x="Road width (m)", y="Road capacity (veh/hour)", col="Road\nclass")
print(p)
mypath <- "plots/TfL_widths.png"
#ggsave(mypath, plot = last_plot())

# Define a non-linear model: Capacity = a * (lanes^b) * (width^c) * (road_class^d)
nls_model <- nlsLM(capacity ~ a * (lanes^b) * (width^c) * (as.numeric(road_class)^d), 
                   data = cap,
                   start = list(a = 100, b = 0.5, c = 0.5, d = 0.2),  # Initial guesses
                   control = list(maxiter = 500))

# Model summary
summary(nls_model)

# Predict and visualize results
cap$predicted_capacity <- predict(nls_model)

# Plot observed vs predicted values
p <- ggplot(cap, aes(x=capacity, y=predicted_capacity, col=class)) +
  geom_abline(slope=1, intercept=0, linetype="dashed", color = "black") +
  geom_point() +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
      labs(x="Actual capacity (veh/hour)", y="Predicted capacity (veh/hour)",
           col="Road\nclass")
print(p)
mypath <- "plots/non_linear_regression_capacity.png"
#ggsave(mypath, plot = last_plot())

params <- data.frame(a=274.29825, b=-0.07768, c=0.90550, d=-0.30351)
write_csv(params, 'outputs/Capacity_Model_Parameters.csv')

##### HIGHWAY WIDTH CHECK #####

width <- read_csv('highway_check_width.csv')
width <- width[,2:ncol(width)]

vals <- data.frame(width[!is.na(width$width),])
vals <- vals[order(vals$width),]

p <- ggplot(width[!is.na(width$width),],
            aes(x=factor(lanes), y=speedlim, col=width)) +
  geom_jitter(size=3) +
  scale_color_viridis(discrete=FALSE, direction=-1) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  labs(x="Number of lanes", y="Speed limit (miles/hour)",
       col="Road\nwidth\n(m)")
print(p)
mypath <- "plots/widths.png"
ggsave(mypath, plot = last_plot())

width_model <- lm(width ~ speedlim + lanes, data = width, na.action = na.omit)

bloop <- data.frame(predict(width_model,
                               newdata = width[is.na(width$width), !(names(width)=='highway_key')]))

width$width[is.na(width$width)] <- predict(width_model,
                                           newdata = width[is.na(width$width), !(names(width)=='highway_key')])
