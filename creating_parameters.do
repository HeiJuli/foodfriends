// Clear everything in memory
clear
clear all

sysuse auto, clear

// Call data set used

use "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\foodfriends-main\data_parametrisation_reduced", clear

//Create diet variable
gen diet=ag07a146
gen diet2="veg" if diet==1
replace diet2="meat" if diet==0
drop diet
rename diet2 diet

//Generate mean of all health parameters - the higher these values, the more emphasis placed on them
egen animal_vars_mean = rowmean( oi18a003 oi18a004 oi18a009 oi18a011 oi18a013 oi18a015 ) //animals
egen health_vars_mean = rowmean( oi18a001 oi18a007 oi18a014 ) //health
egen env_vars_mean = rowmean(oi18a002 oi18a006 oi18a008 oi18a010 oi18a012) //environment

// Generate variables for our parameters
egen index_theta = rowmean( animal_vars_mean health_vars_mean env_vars_mean ) // between 1 and 7 for now
gen index_beta= ag07a013

// Keep observations that have theta and beta
keep if index_beta!=.

// Rescale values

gen theta = -1 +2* ((index_theta-1)/(7-1))
gen beta = (index_beta-1)/(5-1)

// Create alpha as complement of beta

gen alpha=.
replace alpha=1 if beta==0
replace alpha=0.75 if beta==0.25
replace alpha=0.5 if beta==0.5
replace alpha=0.25 if beta==0.75
replace alpha=0 if beta==1

// We have 699 people, of which 22 are veggies

merge 1:1 nomem_encr using "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\foodfriends-main\demographics_2008"
drop if _merge == 2

//Delete variables we dont need anymore

keep nomem_encr nohouse_encr diet theta beta alpha wave yobirth age brutinc netinc brutcat netcat educnodiploma educdiploma educcat _merge

//Save data set
save "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\foodfriends-main\final_data_parameters_with_demographics", replace

export excel "final_data_parameters_with_demographics.csv", firstrow(varlabels) replace

//Plot and save

histogram alpha, normopts(lcolor(green) lwidth(thick)) name(alpha)
graph export "C:\Users\emma.thill\Dropbox\Projects\foodfriends-main\alpha.pdf", as(pdf) name("alpha") replace

histogram beta, normopts(lcolor(green) lwidth(thick)) name(beta)
graph export "C:\Users\emma.thill\Dropbox\Projects\foodfriends-main\beta.pdf", as(pdf) name("beta") replace

histogram theta, normopts(lcolor(green) lwidth(thick)) name(theta)
graph export "C:\Users\emma.thill\Dropbox\Projects\foodfriends-main\theta.pdf", as(pdf) name("theta") replace
