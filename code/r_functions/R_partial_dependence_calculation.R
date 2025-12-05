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

font <- "Times New Roman"

# working directory
setwd("")  # INSERT WORKING DIRECTORY

dcol <- "#0A0A0A"

##### PARAMETER DEFINITIONS #####

# files to be read/written in this script
args <- c("inputs/samSimpTOT_30m.csv",                          #1   input
          "inputs/occupancyRingRoad.csv",                       #2   input
          "inputs/occupancyRingRoad_240123_310323.csv",         #3   input
          "inputs/sensor_lookup.csv",                           #4   input
          "inputs/met_data_010123_311023.csv",                  #5   input
          "outputs/Data_Prepared_Density.csv",                  #6   output
          "outputs/Partial_Dependence_Density_Linear.csv",      #7   output (overwritten by #10)
          "outputs/Fitted_Function_Values_Linear.csv",          #8   output
          "outputs/Fitted_Model_Parameters_Linear.csv",         #9   output
          "outputs/Partial_Dependence_With_Fit_Linear.csv")     #10   output

# T or F: is this the first run with data,
# i.e. the Random Forest model needs fitting?
firstRun <- T
# T or F: does this run need to produce output csv files for the
# exponential model fitting?
writeRun <- T

##### LOAD DATA & PERFORM PRE-PROCESSING #####

data <- read_csv(args[1])
occ <- read_csv(args[2])
occX <- read_csv(args[3])
lookup <- read_csv(args[4])
weather <- read_csv(args[5])

# re-order and then re-name occupancy columns
occX <- occX[,c(1,2,4,3,5,6)]
names(occX) <- names(occ)
occX$occup_LocalTime <- as.character(occX$occup_LocalTime)
occX <- bind_rows(occX,occ)

# join lanes
uniqueSensors <- data.frame(Sensors=unique(occX$occup_sensor))
#write_csv(uniqueSensors, 'inputs/unique_sensors.csv')

# assign half to data
data$half <- ifelse((lubridate::minute(data$timestamp)*60 + lubridate::second(data$timestamp)) < 30*60, 1, 2)

# assign month/day/hour to weather
dateVec <- as.Date(x = integer(0), origin = "1970-01-01")
weather$hour <- NA
for (ii in 1:nrow(weather)) {
  splitted <- strsplit(weather$DateTime[ii], " ")
  dateVec[ii] <- as.Date(splitted[[1]][1],"%d/%m/%Y")
  nearestHr <- splitted[[1]][2]
  splitted <- strsplit(nearestHr, ":")
  weather$hour[ii] <- as.numeric(splitted[[1]][1])
}
weather$date <- dateVec
weather$month <- month(weather$date)
weather$day <- day(weather$date)

# assign month/day/hour/half to occupancy
occX$date <- as_datetime(occX$occup_time)
occX$day <- day(occX$date)
occX$month <- month(occX$date)
occX$hour <- hour(occX$date)
occX$half <- ifelse((lubridate::minute(occX$date)*60 + lubridate::second(occX$date)) < 30*60, 1, 2)

# pre-process data and aggregate by hour
NO2_offset <- min(data$NO2)
data$NO2 <- data$NO2 - NO2_offset
# aggregate
data_h <- data %>%
  dplyr::group_by(month, day, hour, sensorID) %>%
  dplyr::summarise(flow=sum(flowTotal), hum=mean(HUM), temp=mean(TEMP),
            airpres=mean(AIRPRES), no2=mean(NO2))

# pre-process occupancy and aggregate by hour
# join lanes/direction and filter sensors
lookup <- lookup[!is.na(lookup$lanes),]
occX <- occX[occX$occup_sensor %in% c("[SCC]DET003","[SCC]1YHD2"),]
occX <- inner_join(occX, lookup, by="occup_sensor")
ggplot(occX[occX$occup_interval==1,], aes(x=occup_occupancy)) + geom_histogram()
ggplot(occX[occX$occup_interval==5,], aes(x=occup_occupancy)) + geom_histogram()
ggplot(occX[occX$occup_interval==5 & occX$occup_occupancy>=0 & occX$occup_occupancy<=100,], aes(x=occup_occupancy)) + geom_histogram()
occ1 <- occX[occX$occup_interval==1,]
occ5 <- occX[occX$occup_interval==5,]
min(occ1$occup_LocalTime)
max(occ1$occup_LocalTime)
min(occ5$occup_LocalTime)
max(occ5$occup_LocalTime)
# compute density
#occX$occup_occupancy <- occX$occup_occupancy / occX$occup_interval
min(occX$occup_occupancy)
max(occX$occup_occupancy)
length(which(occX$occup_occupancy >= 0 & occX$occup_occupancy <= 100))*100/nrow(occX)
occX <- occX[occX$occup_occupancy >= 0 & occX$occup_occupancy <= 100,]

###
# occX$density_per_lane <- occX$occup_occupancy
# occX$density <- occX$density_per_lane * 2
###

l <- 2 # sensor is 2 meters
X <- occX$occup_occupancy * 1000 # meters in a kilometer
Y <- X/l
occX$density_per_lane <- Y/100 # divide by 100 as occup was given as %
occX$density <- occX$density_per_lane * 2# occX$lanes # multiply by number of lanes
# ggplot(occX, aes(x=occup_time, y=density_per_lane, col=occup_msm)) + geom_point()
# ggplot(occX, aes(x=occup_time, y=occup_msm)) + geom_point()
# aggregate
occ_h <- occX %>%
  dplyr::group_by(month, day, hour, sensorID, occup_sensor) %>%
  dplyr::summarise(density=mean(density), density_per_lane=mean(density_per_lane))
min(occ_h$density_per_lane)
max(occ_h$density_per_lane)
occ_h <- occX %>%
  dplyr::group_by(month, day, hour, sensorID) %>%
  dplyr::summarise(density=mean(density), density_per_lane=mean(density_per_lane))

# pre-process weather and aggregate by hour
# aggregate
weather_h <- weather %>%
  dplyr::group_by(month, day, hour) %>%
  dplyr::summarise(temp_w=mean(`Temp (C)`), airpres_w=mean(`Pressure (hPa)`),
                   rain=mean(`Rain (mm)`), ws_10m=mean(`Wind Speed 10 m (m/s)`),
                   ws_24m=mean(`Wind Speed 24 m (m/s)`), wd_10m=mean(`Wind Direction 10 m (deg.M)`),
                   wd_24m=mean(`Wind Direction 24 m (deg.M)`), pbl=mean(`PBL height (m)`))

# join three data sets
data_h$month <- as.numeric(data_h$month)
data_h$day <- as.numeric(data_h$day)
data_h$hour <- as.numeric(data_h$hour)
#data_h$half <- as.numeric(data_h$half)

occ_h$month <- as.numeric(occ_h$month)
occ_h$day <- as.numeric(occ_h$day)
occ_h$hour <- as.numeric(occ_h$hour)
#occ_h$half <- as.numeric(occ_h$half)

weather_h$month <- as.numeric(weather_h$month)
weather_h$day <- as.numeric(weather_h$day)
weather_h$hour <- as.numeric(weather_h$hour)

dateJoin <- weather[,c("day","month","hour","date")]
dateJoin <- dateJoin[!duplicated(dateJoin),]

d <- inner_join(data_h, occ_h, by=c("day","month","hour","sensorID"))
d <- inner_join(d, weather_h, by=c("day","month","hour"))
d <- inner_join(d, dateJoin, by=c("day","month","hour"))

d$flow_tot <- d$flow * 2 # multiply by number of lanes

# plots
# no2 timeseries
p <- ggplot(d, aes(x=date, y=no2)) +
  geom_line(linewidth=0.4) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  scale_y_continuous(breaks=seq(0,100,10), labels=as.character(seq(0,100,10))) +
  labs(x="Date in 2023", y=expression("NO"[2]*" concentration (ppb)"))
print(p)
if (writeRun) {
  mypath <- "plots/Data_NO2_Timeseries.png"
  ggsave(mypath, plot = last_plot())
}

p <- ggplot(d, aes(x=date, y=flow)) +
  geom_line(linewidth=0.1) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  scale_y_continuous(breaks=seq(0,2500,500), labels=as.character(seq(0,2500,500))) +
  labs(x="Date in 2023", y="Flow (veh/hour/lane)")
print(p)
if (writeRun) {
  mypath <- "plots/Data_Flow_Timeseries.png"
  ggsave(mypath, plot = last_plot())
}

p <- ggplot(d, aes(x=date, y=density_per_lane)) +
  geom_line(linewidth=0.1) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  scale_y_continuous(breaks=seq(0,500,100), labels=as.character(seq(0,500,100))) +
  labs(x="Date in 2023", y="Density (veh/km/lane)")
print(p)
if (writeRun) {
  mypath <- "plots/Occ_Density_Timeseries_Per_Lane_Trimmed_Sensors.png"
  ggsave(mypath, plot = last_plot())
}

p <- ggplot(d, aes(x=density_per_lane)) +
  geom_histogram(fill="black", binwidth=5) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  #scale_y_continuous(breaks=seq(0,300,50), labels=as.character(seq(0,300,50))) +
  labs(x = "Density (veh/km/lane)", y = "Count")
print(p)
if (writeRun) {
  mypath <- "plots/Density_Histogram_Per_Lane_Trimmed_Sensors.png"
  ggsave(mypath, plot = last_plot())
}

p <- ggplot(d, aes(x=density)) +
  geom_histogram(fill="black", binwidth=5) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  #scale_y_continuous(breaks=seq(0,300,50), labels=as.character(seq(0,300,50))) +
  labs(x = "Density (veh/km)", y = "Count")
print(p)
if (writeRun) {
  mypath <- "plots/Density_Histogram_All_Trimmed_Sensors.png"
  ggsave(mypath, plot = last_plot())
}

# FUNDAMENTAL DIAGRAM
d$weekday <- weekdays(d$date, abbr=TRUE)
d$weekend <- ifelse(d$weekday %in% c("Sat","Sun"), 1, 0)
d$timeband <- ifelse(d$hour >= 7 & d$hour <= 9, "AM peak",
                     ifelse(d$hour >= 10 & d$hour <= 10, "Midday",
                            ifelse(d$hour >= 16 & d$hour <= 19, "PM peak",
                            "Night")))
d$timeband <- factor(d$timeband, levels=c("AM peak","Midday","PM peak","Night"))

anoms <- d[d$density_per_lane >= 100 & d$density_per_lane <= 150 & d$flow <= 450,]
anoms2 <- d[d$density_per_lane <= 40,]

funDiag <- d[,c("flow","density","density_per_lane","month","day","hour","weekend")]
#funDiag <- funDiag[!(funDiag$day==1 & funDiag$month==3),]
#funDiag <- funDiag[funDiag$weekend==0,]
#funDiag <- funDiag[funDiag$hour %in% c(7:9,16:19),]
funDiag <- funDiag %>%
  dplyr::group_by(month,day,hour) %>%
  dplyr::mutate(density = mean(density),
                density_per_lane = mean(density_per_lane),
                flow = mean(flow)) %>%
  dplyr::ungroup() %>%
  dplyr::select(density,density_per_lane,flow)
funDiag <- funDiag[!duplicated(funDiag),]

# capacity and critical density
flowSort <- funDiag[order(funDiag$flow),]
flowRank <- ceiling((95/100)*nrow(flowSort))
flow95P <- flowSort$flow[flowRank]
density95P <- flowSort$density_per_lane[flowRank]
cap <- 1470*5/6
#d$capacity <- ifelse(d$sensorID=="SBT",1470,1470*2/3)
d$capacity_per_lane <- cap
cdEst <- d$density_per_lane * d$capacity_per_lane / (d$flow)
mean(cdEst)
median(cdEst)
# critDens <- funDiag[funDiag$flow == max(funDiag$flow),]
# critDens <- min(critDens$density_per_lane)
# critDens
critDens <- round(density95P,1)
critDens
d$critical_density_per_lane <- critDens
d$capacity <- d$capacity_per_lane * 2
d$critical_density <- d$critical_density_per_lane * 2

funDiag2 <- inner_join(occX, data_h[,c("month","day","hour","sensorID","flow")],
                   by=c("month","day","hour","sensorID"))
funDiag2 <- funDiag2[,c("sensorID","occup_sensor","flow","density","density_per_lane","lanes")]

xbreaks <- c(0,50,critDens,100,150,200,250)
xlabels <- c("0","50",deparse(round(critDens,1)),"100","150","200","250")
p <- ggplot() +
  geom_point(data=funDiag, aes(x=density_per_lane, y=flow),
             size=1, shape=21, fill="black", col="black") +
  geom_vline(xintercept=critDens, linetype="dashed", linewidth=1) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=22, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_continuous(breaks=xbreaks, labels=xlabels) +
  scale_y_continuous(breaks=seq(0,1300,150), labels=as.character(seq(0,1300,150))) +
  labs(x = "Density (veh/km/lane)", y = "Flow (veh/hour/lane)")
print(p)
if (writeRun) {
  mypath <- "plots/Fun_Diagram_Final.png"
  ggsave(mypath, plot = last_plot())
}

p <- ggplot() +
  geom_point(data=funDiag, aes(x=density_per_lane, y=flow),
             size=1, shape=21, fill="black", col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=20, family=font),
        panel.grid.major=element_line(color="grey")) +
  # scale_x_continuous(breaks=seq(0,450,50), labels=as.character(seq(0,450,50))) +
  # scale_y_continuous(breaks=seq(0,30,2.5), labels=as.character(seq(0,30,2.5))) +
  labs(x = "Density (veh/km/lane)", y = "Flow (veh/hour/lane)")
print(p)
if (writeRun) {
  mypath <- "plots/Fun_Diagram_Mean_Per_Lane_Weekday.png"
  ggsave(mypath, plot = last_plot())
}

p <- ggplot() +
  geom_point(data=d, aes(x=density_per_lane, y=flow, fill=timeband),
             size=1.5, shape=21, alpha=0.75, col="black")+#, col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=20, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_fill_viridis(discrete=TRUE) +
  scale_x_continuous(breaks=seq(0,450,50), labels=as.character(seq(0,450,50))) +
  scale_y_continuous(breaks=seq(0,1500,200), labels=as.character(seq(0,1500,200))) +
  labs(x = "Density (veh/km/lane)", y = "Flow (veh/hour/lane)", fill="Timeband")
print(p)
if (writeRun) {
  mypath <- "plots/Fun_Diagram_Timeband.png"
  ggsave(mypath, plot = last_plot())
}

p <- ggplot() +
  geom_point(data=d, aes(x=density_per_lane, y=flow, col=factor(weekend)),
             size=1, shape=21, fill="black")+#, col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=20, family=font),
        panel.grid.major=element_line(color="grey")) +
  # scale_x_continuous(breaks=seq(0,450,50), labels=as.character(seq(0,450,50))) +
  # scale_y_continuous(breaks=seq(0,30,2.5), labels=as.character(seq(0,30,2.5))) +
  labs(x = "Density (veh/km/lane)", y = "Flow (veh/hour/lane)", col="Weekend")
print(p)
if (writeRun) {
  mypath <- "plots/Fun_Diagram_Weekend.png"
  ggsave(mypath, plot = last_plot())
}

p <- ggplot() +
  geom_point(data=d, aes(x=density_per_lane, y=flow, fill=sensorID),
             size=1, shape=21, col="black")+#, col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=20, family=font),
        panel.grid.major=element_line(color="grey")) +
  # scale_x_continuous(breaks=seq(0,450,50), labels=as.character(seq(0,450,50))) +
  # scale_y_continuous(breaks=seq(0,30,2.5), labels=as.character(seq(0,30,2.5))) +
  labs(x = "Density (veh/km/lane)", y = "Flow (veh/hour/lane)", fill="Direction")
print(p)
if (writeRun) {
  mypath <- "plots/Fun_Diagram_SBT_NBT_Trimmed_Sensors.png"
  ggsave(mypath, plot = last_plot())
}

p <- ggplot() +
  geom_point(data=d, aes(x=density_per_lane, y=flow, fill=hour),
             size=1.5, shape=21, col="black")+#, col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=20, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_fill_viridis() +
  scale_x_continuous(breaks=seq(0,450,50), labels=as.character(seq(0,450,50))) +
  scale_y_continuous(breaks=seq(0,1200,200), labels=as.character(seq(0,1200,200))) +
  labs(x = "Density (veh/km/lane)", y = "Flow (veh/hour/lane)", fill="Hour")
print(p)
if (writeRun) {
  mypath <- "plots/Fun_Diagram_Hour.png"
  ggsave(mypath, plot = last_plot())
}

p <- ggplot() +
  geom_point(data=d[d$hour %in% c(7:9,16:19),], aes(x=density_per_lane, y=flow, fill=hour),
             size=1.5, shape=21, col="black")+#, col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=20, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_fill_viridis() +
  scale_x_continuous(breaks=seq(0,450,50), labels=as.character(seq(0,450,50))) +
  scale_y_continuous(breaks=seq(0,1200,200), labels=as.character(seq(0,1200,200))) +
  labs(x = "Density (veh/km/lane)", y = "Flow (veh/hour/lane)", fill="Hour")
print(p)
if (writeRun) {
  mypath <- "plots/Fun_Diagram_HourPeak_Coloured.png"
  ggsave(mypath, plot = last_plot())
}

p <- ggplot() +
  geom_point(data=d[d$hour==6,], aes(x=density_per_lane, y=flow,),
             size=1.5, shape=21, col="black", fill="black")+#, col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=20, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_continuous(breaks=seq(0,450,50), labels=as.character(seq(0,450,50))) +
  scale_y_continuous(breaks=seq(0,1200,200), labels=as.character(seq(0,1200,200))) +
  xlim(c(0,200)) +
  ylim(c(0,1300)) +
  labs(x = "Density (veh/km/lane)", y = "Flow (veh/hour/lane)")
print(p)
if (writeRun) {
  mypath <- "plots/Fun_Diagram_Hour6.png"
  ggsave(mypath, plot = last_plot())
}

p <- ggplot() +
  geom_point(data=d, aes(x=density_per_lane, y=flow, col=sensorID),
             size=1, shape=21, fill="black")+#, col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=20, family=font),
        panel.grid.major=element_line(color="grey")) +
  # scale_x_continuous(breaks=seq(0,450,50), labels=as.character(seq(0,450,50))) +
  # scale_y_continuous(breaks=seq(0,30,2.5), labels=as.character(seq(0,30,2.5))) +
  labs(x = "Density (veh/km)", y = "Flow (veh/hour)", col="Direction")
print(p)
if (writeRun) {
  mypath <- "plots/Fun_Diagram_SBT_NBT_Per_Lane.png"
  ggsave(mypath, plot = last_plot())
}

uniqueSens <- unique(funDiag2$occup_sensor)
for (ii in 1:length(uniqueSens)) {
  sens <- uniqueSens[ii]
  plotDat <- funDiag2[funDiag2$occup_sensor==sens,]
  plotDat <- plotDat[!duplicated(plotDat),]
  
  p <- ggplot() +
    geom_point(data=plotDat, aes(x=density_per_lane, y=flow),
               size=1, shape=21, fill="black", col="black") +
    theme(panel.background = element_rect(fill="white", colour="grey"),
          text=element_text(size=20, family=font),
          panel.grid.major=element_line(color="grey")) +
    # scale_x_continuous(breaks=seq(0,450,50), labels=as.character(seq(0,450,50))) +
    # scale_y_continuous(breaks=seq(0,30,2.5), labels=as.character(seq(0,30,2.5))) +
    labs(x = "Density (veh/km/lane)", y = "Flow (veh/hour/lane)")
  if (writeRun) {
    print(p)
    mypath <- paste0("plots/Sensors/LANE_Fun_Diagram_", sens, ".png")
    ggsave(mypath, plot = last_plot())
  }
  
  plotDat <- plotDat %>%
    dplyr::group_by(density_per_lane) %>%
    dplyr::mutate(density_per_lane_mean = mean(density_per_lane),
                  flow_mean = mean(flow)) %>%
    dplyr::ungroup() %>%
    dplyr::select(density_per_lane_mean, flow_mean, occup_sensor)
  plotDat <- plotDat[!duplicated(plotDat),]
  
  p <- ggplot() +
    geom_point(data=plotDat, aes(x=density_per_lane_mean, y=flow_mean),
               size=1, shape=21, fill="black", col="black") +
    theme(panel.background = element_rect(fill="white", colour="grey"),
          text=element_text(size=20, family=font),
          panel.grid.major=element_line(color="grey")) +
    # scale_x_continuous(breaks=seq(0,450,50), labels=as.character(seq(0,450,50))) +
    # scale_y_continuous(breaks=seq(0,30,2.5), labels=as.character(seq(0,30,2.5))) +
    labs(x = "Density (veh/km/lane)", y = "Flow (veh/hour/lane)")
  if (writeRun) {
    print(p)
    mypath <- paste0("plots/Sensors/LANE_Mean_Fun_Diagram_", sens, ".png")
    ggsave(mypath, plot = last_plot())
  }
}

meansNO2 <- d %>%
  dplyr::group_by(date, month, day, hour) %>%
  dplyr::summarise(no2 = mean(no2), density = mean(density), flow_tot = mean(flow_tot))

# NO2 VS DENSITY
p <- ggplot() +
  geom_point(data=meansNO2, aes(x=density, y=no2), size=0.75, shape=21,
             fill="black", col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=22, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_continuous(breaks=seq(0,1000,50), labels=as.character(seq(0,1000,50))) +
  scale_y_continuous(breaks=seq(0,100,10), labels=as.character(seq(0,100,10))) +
  labs(x = "", y = expression("NO"[2]*" concentration (ppb)"))
print(p)
if (writeRun) {
  mypath <- "plots/no2_vs_density_Final.png"
  ggsave(mypath, plot = last_plot())
}

# NO2 VS FLOW
p <- ggplot() +
  geom_point(data=meansNO2, aes(x=flow_tot, y=no2), size=0.75, shape=21,
             fill="black", col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=22, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_continuous(breaks=seq(0,3000,250), labels=as.character(seq(0,3000,250))) +
  scale_y_continuous(breaks=seq(0,100,10), labels=as.character(seq(0,100,10))) +
  labs(x = "", y = expression("NO"[2]*" concentration (ppb)"))
print(p)
if (writeRun) {
  mypath <- "plots/no2_vs_flow_Final.png"
  ggsave(mypath, plot = last_plot())
}

# TEMP TIMESERIES
# data
p <- ggplot(d, aes(x=date, y=temp)) +
  geom_line(linewidth=0.1) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  scale_y_continuous(breaks=seq(0,35,5), labels=as.character(seq(0,35,5))) +
  labs(x="Date in 2023", y="Temperature (\u00B0C)")
print(p)
if (writeRun) {
  mypath <- "plots/Data_Temp_Timeseries.png"
  ggsave(mypath, plot = last_plot())
}
# weather
p <- ggplot(d, aes(x=date, y=temp_w)) +
  geom_line(linewidth=0.1) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  scale_y_continuous(breaks=seq(0,35,5), labels=as.character(seq(0,35,5))) +
  labs(x="Date in 2023", y="Temperature (\u00B0C)")
print(p)
if (writeRun) {
  mypath <- "plots/Weather_Temp_Timeseries.png"
  ggsave(mypath, plot = last_plot())
}

# PBL HEIGHT
p <- ggplot(d, aes(x=date, y=pbl)) +
  geom_line(linewidth=0.1) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  scale_y_continuous(breaks=seq(0,2200,200), labels=as.character(seq(0,2200,200))) +
  labs(x="Date in 2023", y="Planetary boundary\nlayer height (m)")
print(p)
if (writeRun) {
  mypath <- "plots/Weather_PBL_Timeseries.png"
  ggsave(mypath, plot = last_plot())
}

# RAIN
p <- ggplot(d, aes(x=date, y=rain)) +
  geom_line(linewidth=0.1) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  #scale_y_continuous(breaks=seq(0,2200,200), labels=as.character(seq(0,2200,200))) +
  labs(x="Date in 2023", y="Rainfall (mm)")
print(p)
if (writeRun) {
  mypath <- "plots/Weather_Rain_Timeseries.png"
  ggsave(mypath, plot = last_plot())
}

# WIND SPEED 10m
p <- ggplot(d, aes(x=date, y=ws_10m)) +
  geom_line(linewidth=0.1, col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  scale_y_continuous(breaks=seq(0,15,2), labels=as.character(seq(0,15,2))) +
  labs(x="Date in 2023", y=bquote("Wind speed 10m above ground ("~m/s^2*")"))
print(p)
if (writeRun) {
  mypath <- "plots/Weather_WindSpeed10m_Timeseries.png"
  ggsave(mypath, plot = last_plot())
}

# WIND SPEED 24m
p <- ggplot(d, aes(x=date, y=ws_24m)) +
  geom_line(linewidth=0.1, col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  scale_y_continuous(breaks=seq(0,15,2), labels=as.character(seq(0,15,2))) +
  labs(x="Date in 2023", y=bquote("Wind speed 24m above ground ("~m/s^2*")"))
print(p)
if (writeRun) {
  mypath <- "plots/Weather_WindSpeed24m_Timeseries.png"
  ggsave(mypath, plot = last_plot())
}

# WIND DIRECTION 10m
p <- ggplot(d, aes(x=date, y=wd_10m)) +
  geom_line(linewidth=0.1, col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  scale_y_continuous(breaks=seq(0,400,50), labels=as.character(seq(0,400,50))) +
  labs(x="Date in 2023", y=bquote("Wind direction 10m above ground"))
print(p)
if (writeRun) {
  mypath <- "plots/Weather_WindDirection10m_Timeseries.png"
  ggsave(mypath, plot = last_plot())
}

# WIND DIRECTION 24m
p <- ggplot(d, aes(x=date, y=wd_24m)) +
  geom_line(linewidth=0.1, col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  scale_y_continuous(breaks=seq(0,400,50), labels=as.character(seq(0,400,50))) +
  labs(x="Date in 2023", y=bquote("Wind direction 24m above ground"))
print(p)
if (writeRun) {
  mypath <- "plots/Weather_WindDirection24m_Timeseries.png"
  ggsave(mypath, plot = last_plot())
}

d$day_julian <- lubridate::yday(d$date)
d$date <- as.POSIXct(d$date)

# estimated speed
d$speed <- d$flow / d$density_per_lane
mean(d$speed[!is.infinite(d$speed)])
median(d$speed[!is.infinite(d$speed)])

max(d$speed[!is.infinite(d$speed)])

p <- ggplot(d[!is.infinite(d$speed),], aes(x=speed)) +
  geom_histogram()
print(p)

#write_csv(d, 'R Outputs/Data_Processed_Trimmed_Sensors.csv')

##### PREPARE DATA & TRAIN RANDOM FOREST MODELS #####

if (firstRun) {
  set.seed(42)
  Y <- 'no2'
  x <- 'density'
  X <- c('temp', 'airpres', 'ws_10m', 'wd_10m', 'ws_24m', 'wd_24m', 'hum')
  #X <- c('flow_tot', 'temp', 'airpres', 'ws_10m', 'wd_10m', 'ws_24m', 'wd_24m', 'hum', 'rain', 'temp_w', 'airpres_w')
  #X <- c('temp_w', 'airpres_w', 'temp', 'airpres', 'ws_10m', 'wd_10m', 'hum', 'flow')
  tVars <- c('month', 'day', 'hour', 'half', 'day_julian', 'date')
  #tVars <- c('date')
  
  dat <- d[,names(d) %in% c(Y,x,X,tVars)]
  dat <- dat %>%
    dplyr::group_by(date,month,day,hour,day_julian) %>%
    dplyr::summarise(temp=mean(temp), airpres=mean(airpres), ws_10m=mean(ws_10m), wd_10m=mean(wd_10m),
                     ws_24m=mean(ws_24m), wd_24m=mean(wd_24m),
                     hum=mean(hum), no2=mean(no2), density=mean(density)) %>%
    # dplyr::summarise(temp=mean(temp),
    #           airpres=mean(airpres), ws_10m=mean(ws_10m), wd_10m=mean(wd_10m),
    #           ws_24m=mean(ws_24m), wd_24m=mean(wd_24m), hum=mean(hum),
    #           rain=sum(rain), flow_tot=mean(flow_tot), no2=mean(no2), density=mean(density)) %>%
    dplyr::ungroup() %>%
    dplyr::select(date,month,day,hour,day_julian,temp,airpres,ws_10m,wd_10m,ws_24m,wd_24m,hum,no2,density)
  
  ##### PREPARE DATA AND TRAIN MONTE-CARLO MODELS #####
  dataPrep <- rmw_prepare_data(dat, 'no2', fraction = 0.8)
  train <- dataPrep[dataPrep$set=="training",]
  test <- dataPrep[dataPrep$set=="testing",]
  #toAvg <- c(X[X!=x])#, tVars[tVars!="date"])
  
  # mod <- rmw_train_model(dataPrep, c(X,x,tVars), n_trees=600, mtry=3)
  mod <- rmw_train_model(dataPrep, c(X,x,'date','month','day','hour','day_julian'), n_trees=600, mtry=3)
  pred <- rmw_predict_the_test_set(mod, dataPrep)
  corTest <- cor(pred$value, pred$value_predict)
  corTest
  
  varImps <- data.frame(Feature = names(mod$variable.importance),
                       Importance = as.numeric(mod$variable.importance))
  varImps <- varImps[order(varImps$Importance, decreasing=TRUE),]
  #write_csv(varImps, 'R Outputs/Importance_of_Features_in_RF_Model_Reduced.csv')
  
  pDat <- dataPrep
  pDep <- rep(0,nrow(pDat))
  for (ii in 1:nrow(pDat)) {
    m <- pDat
    # repeat density value at row ii
    m$density <- m$density[ii]
    # predict model
    out <- rmw_predict_the_test_set(mod, m)
    # assign partial dependency of no2 on density ii as the mean predicted value
    pDep[ii] <- mean(out$value_predict)
  }
  
  PD <- data.frame(Actual=pDat$value, Partial_Dep=pDep, Density=pDat$density)
  # PD <- PD %>%
  #   group_by(Density) %>%
  #   dplyr::mutate(Mean_Partial = mean(Partial)) %>%
  #   ungroup()
  
  ### write files
  write_csv(dataPrep, args[6])
  write_csv(PD, args[7])
} else {
  dataPrep <- read_csv(args[6])
  #PD <- read_csv(args[7])
  fittedVals <- read_csv(args[8])
  params <- read_csv(args[9])
  PD_with_fit <- read_csv(args[10])
}

##### PLOT PARTIAL DEPENDENCE AND FIT EXPONENTIAL MODELS #####

PD <- PD_with_fit
PD$Partial <- PD$Partial_Dep

### OFFSET TO ZERO ###
minRes <- min(PD$Partial_Dep)
minRes
PD$Partial <- PD$Partial_Dep - minRes

prop <- PD$Partial_Dep / PD$Partial
prop <- prop[!is.infinite(prop)]
median(prop)
mean(prop)

p <- ggplot(PD, aes(x=Density, y=Partial)) +
  geom_point(size=1) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_continuous(breaks=seq(0,500,50), labels=as.character(seq(0,500,50))) +
  scale_y_continuous(breaks=seq(0,40,0.5), labels=as.character(seq(0,40,0.5))) +
  #ylim(c(0,19)) +
  labs(x = "Density (veh/km)", y = expression("NO"[2]*" partial dependence (ppb)"))
print(p)
if (writeRun) {
  mypath <- "plots/Partial_Dependence.png"
  ggsave(mypath, plot = last_plot())
}

p <- ggplot() +
  geom_point(data=PD, aes(x=Density, y=Actual-minRes), size=0.75, col="black") +
  geom_point(data=PD, aes(x=Density, y=Partial), size=0.75, col="red") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey")) +
  #scale_x_continuous(breaks=seq(0,250,25), labels=as.character(seq(0,250,25))) +
  #scale_y_continuous(breaks=seq(15,40,2.5), labels=as.character(seq(15,40,2.5))) +
  labs(x = "Density (veh/km)", y = expression("Offset NO"[2]*" concentration (ppb)"))
print(p)
if (writeRun) {
  mypath <- "plots/Partial_and_Actual_NO2_vs_Density.png"
  ggsave(mypath, plot = last_plot())
}

varImps$Importance <- varImps$Importance * 100 / sum(varImps$Importance)
varImps$Feature[varImps$Feature=="temp"] <- "temperature"
varImps$Feature[varImps$Feature=="hum"] <- "humidity"
varImps$Feature[varImps$Feature=="airpres"] <- "air_pressure"
varImps$Feature <- factor(varImps$Feature, levels=varImps$Feature)
# importance of variables
p <- ggplot(varImps, aes(x=Feature, y=Importance)) +
  geom_bar(stat="identity", fill="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=23, family=font),
        panel.grid.major=element_line(color="grey"),
        axis.text.x=element_text(angle=90, vjust=0.5, hjust=1)) +
  scale_y_continuous(breaks=seq(0,100,2.5), labels=as.character(seq(0,100,2.5))) +
  labs(x = "Feature", y = "Importance (%)")
print(p)
if (writeRun) {
  mypath <- "plots/Feature_Importance.png"
  ggsave(mypath, plot = last_plot())
}

p <- ggplot() +
  geom_point(data=pred, aes(x=value, y=value_predict), size=1, shape=21,
             col="black", fill="black") +
  geom_abline(linewidth=0.75, linetype="dashed") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=22, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_continuous(breaks=seq(0,80,5), labels=as.character(seq(0,80,5))) +
  scale_y_continuous(breaks=seq(0,80,5), labels=as.character(seq(0,80,5))) +
  #coord_fixed() +
  labs(x = expression("Actual NO"[2]*" concentration (ppb)"), y = expression("Predicted NO"[2]*" concentration (ppb)"))
print(p)
if (writeRun) {
  mypath <- "plots/Pred_Vs_Actual_NO2.png"
  ggsave(mypath, plot = last_plot())
}

cor(pred$value, pred$value_predict)

crit <- unique(d$CriticalDensity)
PD$CriticalDensity <- crit
cap <- unique(d$capacity)
PD$Capacity <- cap

# BPR #
speedlim <- 64.4
PD$TT <- (1/speedlim) * (1 + 0.15*(PD$Density/PD$CriticalDensity)^4)
PD$TT <- PD$TT * 60 # hours to minutes
p <- ggplot() +
  geom_point(data=PD, aes(x=Density, y=TT), size=1, col="black") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=22, family=font),
        panel.grid.major=element_line(color="grey")) +
  ylim(c(0,20)) +
  scale_x_continuous(breaks=seq(0,500,50), labels=as.character(seq(0,500,50))) +
  scale_y_continuous(breaks=seq(0,50,2), labels=as.character(seq(0,50,2))) +
  labs(x = "Density (veh/km)", y = expression("Travel time (mins/km)"))
print(p)
if (writeRun) {
  mypath <- "plots/Travel_Time_Final.png"
  ggsave(mypath, plot = last_plot())
}

##### FIT MODEL (LINEAR) #####
PD <- PD[order(PD$Density),]
frange <- seq(240,250,0.1)
inds <- which((PD$Density >= min(frange)) & (PD$Density <= max(frange)))
rSquaredVec <- numeric(length(inds))
mVec <- numeric(length(inds))
intercept <- 0
for (ii in 1:length(inds)) {
  ind <- inds[ii]
  threshLoop <- PD$Density[ind]
  no2Loop <- PD$Partial[ind]
  grad <- no2Loop/threshLoop
  # ind <- which(abs(PD$Density-threshLoop)==min(abs(PD$Density-threshLoop)))
  # val <- PD$Partial[ind]
  # linModLoop <- lm(I(Partial - intercept) ~ 0 + Density,
  #                  data = PD[PD$Density<=threshLoop,])
  yPred <- grad*PD$Density[PD$Density<=threshLoop]
  ssRes <- sum((PD$Partial[PD$Density<=threshLoop] - yPred)^2)
  ssTot <- sum((PD$Partial[PD$Density<=threshLoop]-mean(PD$Partial[PD$Density<=threshLoop]))^2)
  rSquaredLoop <- 1 - ssRes/ssTot
  rSquaredVec[ii] <- rSquaredLoop
  mVec[ii] <- grad
}
rSquared <- max(rSquaredVec)
rSquared
bestInds <- which(rSquaredVec==rSquared)
bestInd <- max(bestInds)
thresh <- PD$Density[inds[bestInd]]
thresh
m <- mVec[bestInd]
m

# linMod <- lm(I(Partial - intercept) ~ 0 + Density, data = PD[PD$Density<=thresh,])
# m <- as.numeric(linMod$coefficients)

m <- params$m
thresh <- params$thresh

x0 <- seq(0,450,0.01)
y0 <- m*x0
testDf <- data.frame(x = x0, y = y0)
yPred <- m*PD$Density
p <- ggplot() +
  geom_point(data=PD, aes(x=Density, y=Partial), size=1) +
  geom_line(data=testDf, aes(x=x, y=y), linewidth=1) +
  geom_vline(xintercept=thresh, linetype="dashed", col="black", linewidth=0.75) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=22, family=font),
        panel.grid.major=element_line(color="grey")) +
  scale_x_continuous(breaks=seq(0,500,50), labels=as.character(seq(0,500,50))) +
  scale_y_continuous(breaks=seq(0,20,0.5), labels=as.character(seq(0,20,0.5))) +
  #ylim(c(0,4)) +
  labs(x = "Density (veh/km)", y = expression("NO"[2]*" partial dependence (ppb)"))
print(p)
if (writeRun) {
  mypath <- "plots/PD_with_Fit.png"
  ggsave(mypath, plot = last_plot())
}

speed <- 17.7
PD$Pred <- yPred
fittedVals <- data.frame(Density=x0, NO2=m*x0)
params <- data.frame(m=m, thresh=thresh,
                     criticalDensity=crit, capacity=cap,
                     speed=speed, rSquared=rSquared)

##########
if (writeRun) {
  write_csv(fittedVals, args[8])
  write_csv(params, args[9])
  write_csv(PD, args[10])
} else {
  fittedVals <- read_csv(args[8])
  params <- read_csv(args[9])
  PD <- read_csv(args[10])
  varImps <- read_csv('outputs/Importance_of_Features_in_RF_Model.csv')
}

##### ALPHA AND BETA ####
# Diagnostic function to check input data
validate_input <- function(volume, capacity, observed_travel_times) {
  cat("Input Data Validation:\n")
  cat("Volume/Capacity Ratios:", volume/capacity, "\n")
  cat("Volumes:", volume, "\n")
  cat("Capacities:", capacity, "\n")
  cat("Observed Travel Times:", observed_travel_times, "\n")
}
validate_input(d$flow_tot, d$capacity, observed_travel_times)

estimate_bpr_parameters <- function(volume, capacity, link_length, observed_travel_times, free_flow_speed) {
  # Prepare data
  V_C_ratio <- volume / capacity
  
  # Calculate free-flow travel time
  t0 <- link_length / free_flow_speed
  
  # BPR model function
  bpr_model <- function(params, V_C) {
    alpha <- params[1]
    beta <- params[2]
    t0 * (1 + alpha * (V_C)^beta)
  }
  
  # Objective function with sum of squared errors
  objective <- function(params) {
    # Enforce parameter constraints within function
    if (params[1] < 0 || params[1] > 1 || params[2] < 2 || params[2] > 5) {
      return(Inf)  # Penalize out-of-range parameters
    }
    
    predicted_times <- bpr_model(params, V_C_ratio)
    sum((predicted_times - observed_travel_times)^2)
  }
  
  # Multiple initial parameter guesses to avoid local minima
  initial_guesses <- list(
    c(0.15, 4.0),   # Conservative BPR typical values
    c(0.5, 3.0),    # Mid-range estimate
    c(0.8, 2.5)     # Higher alpha, lower beta
  )
  
  # Store results for comparison
  results <- lapply(initial_guesses, function(init_params) {
    optim(
      par = init_params,
      fn = objective,
      method = "Nelder-Mead",  # Suitable for non-linear optimization
      control = list(maxit = 1000)
    )
  })
  
  # Select the result with lowest objective function value
  best_result <- results[[which.min(sapply(results, `[[`, "value"))]]
  
  return(list(
    alpha = best_result$par[1],
    beta = best_result$par[2],
    value = best_result$value,
    convergence = best_result$convergence
  ))
}

# t0 <- rep(100/speedlim, nrow(d))
# observed_travel_times <- rep(100/speed, nrow(d))

# Example usage
volume <- d$flow_tot
capacity <- d$capacity
V_C_ratio <- volume/capacity
link_length <- 1 # kilometers (ARBITRARY)
observed_travel_times <- rep(link_length/speed, nrow(d))
free_flow_speed <- speedlim  # km/hour

# Estimate parameters
results <- estimate_bpr_parameters(
  volume, 
  capacity, 
  link_length, 
  observed_travel_times, 
  free_flow_speed
)

# Print results with explanatory context
cat("BPR Parameter Estimation Results:\n")
cat("Alpha (Congestion Impact Factor):", round(results$alpha, 4), 
    "(Recommended Range: 0.15 - 0.75)\n")
cat("Beta (Congestion Non-linearity):", round(results$beta, 4), 
    "(Recommended Range: 2.0 - 5.0)\n")
cat("Optimization Convergence:", results$convergence, 
    "(0 indicates successful convergence)\n")

alpha <- round(results$alpha, 4)
beta <- round(results$beta, 4)
TT_pred <- bpr_model(c(alpha,beta), V_C_ratio)

ttRes <- sum((observed_travel_times - TT_pred)^2)
ttTot <- sum((observed_travel_times-mean(observed_travel_times))^2)
rSquaredBPR <- 1 - ttRes/ttTot
rSquaredBPR