library(tidyverse)
library(plyr)
library(dplyr)
library(ggplot2)
library(data.table)
library(viridis)
library(lubridate)
library(reshape2)
library(RColorBrewer)
library(rmweather)
library(extrafont)

font_import()
loadfonts(device = "win")

font <- "Times New Roman"

dcol <- "#0A0A0A"

# working directory
setwd("")  # INSERT WORKING DIRECTORY
inputDir <- ("outputs")

edges <- read_csv("MATLAB Inputs/edges.csv")

flowsUE <- read_csv(paste0(inputDir, "/UE_flow_out.csv"), col_names=FALSE)
flowsPO <- read_csv(paste0(inputDir, "/PO_flow_out.csv"), col_names=FALSE)
no2UE <- read_csv(paste0(inputDir, "/UE_NO2_ugm3_out.csv"), col_names=FALSE)
no2PO <- read_csv(paste0(inputDir, "/PO_NO2_ugm3_out.csv"), col_names=FALSE)

no2UE <- no2UE * 9.034974
no2PO <- no2PO * 9.034974

flowDiff <- flowsPO - flowsUE
no2Diff <- no2PO - no2UE

N <- nrow(flowsUE)
km <- edges$length[1:N]/1000

no2km <- data.frame(UE = no2UE/km, PO = no2PO/km)
names(no2km) <- c("UE","PO")
no2km <- reshape2::melt(no2km)

no2UE_km <- no2UE/km
no2PO_km <- no2PO/km
# TT_UE_km <- read_csv(paste0(inputDir, "/TT_UE_km.csv"))[,2]
# TT_PO_km <- read_csv(paste0(inputDir, "/TT_PO_km.csv"))[,2]

no2Diff_km <- no2PO_km - no2UE_km
#TTDiff_km <- TT_PO_km - TT_UE_km

### ORDER ROAD INDICES BY FLOW DIFFS ###

no2Diff <- data.frame(no2Diff[order(flowDiff$X1),])
names(no2Diff) <- "X1"
no2Diff$ind <- 1:nrow(no2Diff)

no2Diff_km <- data.frame(no2Diff_km[order(flowDiff$X1),])
names(no2Diff_km) <- "X1"
no2Diff_km$ind <- 1:nrow(no2Diff_km)

flowDiff <- data.frame(flowDiff[order(flowDiff$X1),])
names(flowDiff) <- "X1"
flowDiff$ind <- 1:nrow(flowDiff)

### FLOW PLOTS ###

p <- ggplot(flowDiff, aes(x=ind, y=X1)) +
  geom_bar(stat="identity", col="black") +
  guides(col="none") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey"),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  scale_y_continuous(breaks=seq(-5000,5000,250), labels=as.character(seq(-5000,5000,250))) +
  labs(x="Road index", y="Change in flow (veh/hour)")
print(p)
mypath <- "R Plots/FlowDiffs_2025.png"
ggsave(mypath, plot = last_plot())

length(which(flowDiff$X1 == 0))
length(which(flowDiff$X1 > 0))
length(which(flowDiff$X1 < 0))

length(which(flowDiff$X1 == 0))/nrow(flowDiff)*100
length(which(flowDiff$X1 > 0))/nrow(flowDiff)*100
length(which(flowDiff$X1 < 0))/nrow(flowDiff)*100

### NO2 PLOTS ###

p <- ggplot(no2Diff, aes(x=ind, y=X1)) +
  geom_bar(stat="identity", col="black") +
  guides(col="none") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey"),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  scale_y_continuous(breaks=seq(-30,30,5), labels=as.character(seq(-30,30,5))) +
  labs(x="Road index", y=expression("Change in NO "[2]*" concentration ("*mu*"g/m"^3*")"))
print(p)
mypath <- "R Plots/NO2Diffs_2025.png"
ggsave(mypath, plot = last_plot())

length(which(no2Diff$X1 == 0))/nrow(flowDiff)*100
length(which(no2Diff$X1 > 0))/nrow(flowDiff)*100
length(which(no2Diff$X1 < 0))/nrow(flowDiff)*100

no2 <- data.frame(UE = no2UE, PO = no2PO)
names(no2) <- c("UE","PO")
no2 <- reshape2::melt(no2)

length(which(no2Diff$X1 == 0))
length(which(no2Diff$X1 == 0))/nrow(no2Diff)*100

length(which(no2Diff$X1 > 0))
length(which(no2Diff$X1 > 0))/nrow(no2Diff)*100

length(which(no2Diff$X1 < 0))
length(which(no2Diff$X1 < 0))/nrow(no2Diff)*100

length(which(no2PO$X1 < no2UE$X1))*100/nrow(no2UE)
length(which(no2PO$X1 > no2UE$X1))*100/nrow(no2UE)
length(which(no2PO$X1 <= no2UE$X1))*100/nrow(no2UE)

p <- ggplot(no2km[!is.nan(no2km$value),], aes(x=value, fill=variable)) +
  geom_histogram(binwidth=0.1, alpha=.75) +
  #geom_vline(xintercept=50, linetype="dashed", linewidth=1) +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=20, family=font),
        panel.grid.major=element_line(color="grey"),
        legend.position=c(.77,.5)) +
  scale_x_continuous(breaks=seq(0,350,25), labels=as.character(seq(0,350,25))) +
  scale_y_continuous(breaks=seq(0,150,25), labels=as.character(seq(0,150,25))) +
  scale_fill_manual(values=c('#940B13','black')) +
  labs(x=expression("Modelled NO "[2]*" per unit of road ("*mu*"g/m"^3*"/km)"), y="Count",
       fill="Scenario")
print(p)
mypath <- "R Plots/NO2kmHist.png"
ggsave(mypath, plot = last_plot())

p <- ggplot(no2Diff_km[is.finite(no2Diff_km$X1),], aes(x=ind, y=X1)) +
  geom_bar(stat="identity", col="black") +
  guides(col="none") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=16, family=font),
        panel.grid.major=element_line(color="grey"),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  scale_y_continuous(breaks=seq(-100,100,25), labels=as.character(seq(-100,100,25))) +
  labs(x="Road index", y=expression("Change in NO "[2]*" ("*mu*"g/m"^3*")"))
print(p)
mypath <- "R Plots/NO2Diffs_km_2025.png"
ggsave(mypath, plot = last_plot())

length(which(no2Diff_km$X1 == 0))/nrow(no2Diff_km)*100
length(which(no2Diff_km$X1 > 0))/nrow(no2Diff_km)*100
length(which(no2Diff_km$X1 < 0))/nrow(no2Diff_km)*100

TTDiff_km <- data.frame(TTDiff_km[order(TTDiff_km[,1]),])
names(TTDiff_km) <- "X1"
TTDiff_km$ind <- 1:nrow(TTDiff_km)

p <- ggplot(TTDiff_km, aes(x=ind, y=X1)) +
  geom_bar(stat="identity", col="black") +
  guides(col="none") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=20, family=font),
        panel.grid.major=element_line(color="grey"),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  scale_y_continuous(breaks=seq(-50,50,5), labels=as.character(seq(-50,50,5))) +
  labs(x="", y="Change in travel time (km/hour)")
print(p)
mypath <- "RPlots/TTDiffs_km_2025.png"
#ggsave(mypath, plot = last_plot())

############ CAPACITIES ###########
ratUE <- flowsUE/edges$capacity
ratPO <- flowsPO/edges$capacity

UEexceedsCap <- edges[flowsUE > edges$capacity,]
POexceedsCap <- edges[flowsPO > edges$capacity,]

ratUE[flowsUE > edges$capacity]
ratPO[flowsPO > edges$capacity]

mean(ratUE$X1)
mean(ratPO$X1)

# rats <- data.frame("UE" = ratUE$X1, "PO" = ratPO$X1)
rats <- data.frame(X1 = ratPO$X1 - ratUE$X1)
rats <- data.frame(rats[order(rats$X1),])
names(rats) <- "X1"
rats$ind <- 1:nrow(rats)
#rats <- reshape2::melt(rats, id.vars="ind")

p <- ggplot(rats, aes(x=ind, y=X1*100)) +
  geom_bar(stat="identity", col="black") +
  guides(col="none") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=20, family=font),
        panel.grid.major=element_line(color="grey"),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  scale_y_continuous(breaks=seq(-200,200,25), labels=as.character(seq(-200,200,25))) +
  labs(x="", y="Change in flow/capacity (%)")
print(p)
mypath <- "RPlots/ratioDiffs_2025.png"
#ggsave(mypath, plot = last_plot())

increasedFlow <- edges[flowsPO > flowsUE,]

############ NET CHANGES ###########
sum(no2UE)
sum(no2PO)
sum(no2UE) - sum(no2PO)
sum(flowsUE)
sum(flowsPO)
sum(flowsUE) - sum(flowsPO)

############ PATRICIO ############

veh <- read_csv("R Inputs/vehiclesBreakDown.csv")

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
  #scale_fill_brewer(palette = "Greys") +
  theme(panel.background = element_rect(fill="white", colour="grey"),
        text=element_text(size=18, family=font),
        panel.grid.major=element_line(color="grey"),
        axis.text.x=element_text(angle=90, vjust=0.5, hjust=1)) +
  expand_limits(x= c(0, 13)) +
  #scale_y_continuous(breaks=seq(0,1300000,250000), labels=c("0","250K","500K",
                                                           #"750K","1M","1.25M")) +
  labs(x="Combined engine and fuel type", y="Log number of vehicles", fill="Age")
#p <- edit_colors(p, desaturate)
print(p)
mypath <- "RPlots/vehStacks_Summer_Log.png"
ggsave(mypath, plot = last_plot(), width = 8, height = 6, units = "in")

