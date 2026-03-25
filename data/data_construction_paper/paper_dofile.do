// Clear everything in memory
clear
clear all

sysuse auto, clear

// Call data set used

use "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\Data\Final_paper_data\oi18a_EN_1.0p.dta", clear

//Create diet variable
gen diet=oi18a016
gen diet2="veg" if diet==1
replace diet2="meat" if diet==2
drop diet
rename diet2 diet

tab diet
// We have 5742 people of which 69 are veggies (1.2%)

//Generate mean of all health parameters - the higher these values, the more emphasis placed on them
egen animal_vars_mean = rowmean( oi18a003 oi18a004 oi18a009 oi18a011 oi18a013 oi18a015 ) //animals
egen health_vars_mean = rowmean( oi18a001 oi18a007 oi18a014 ) //health
egen env_vars_mean = rowmean(oi18a002 oi18a006 oi18a008 oi18a010 oi18a012) //environment

// Generate variables for our parameters
egen index_theta = rowmean( animal_vars_mean health_vars_mean env_vars_mean ) // between 1 and 7 for now

// Rescale values

gen theta = -1 +2* ((index_theta-1)/(7-1))

keep nomem_encr theta diet

// Subsection 4.4.2

tab diet

// Subsection 4.4.3.

sum theta

// Save

save "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\Data\Final_paper_data\theta_diet.dta", replace

*Continue here with alpha - Section 4.4.4

//Collectivism data (LISS)

use "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\Data\Collectivism\LISS\ed11a_EN_1.0p.dta", clear 

keep nomem_encr ed11a020

rename ed11a020 index_alpha

gen alpha = (index_alpha-1)/(7-1)

//Delete variables we dont need anymore

keep nomem_encr alpha

label variable alpha "Self-identity weight (alpha)"

//Statistics

sum alpha

// Save

save "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\Data\Final_paper_data\alpha.dta", replace

*Continue here with rho - Section 4.4.5

// Call data set used

use "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\Data\Final_paper_Data\su19a_EN_1.0p.dta", clear

keep nomem_encr su19a047 su19a046

rename su19a047 index_rho
rename su19a046 diet_veggie

replace index_rho = 1 if index_rho == 88

gen rho = (index_rho-1)/(4-1)
//total of 2506 responses

save "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\Data\Final_paper_data\rho.dta", replace

// Merge (1) theta and veg with demographics for 2000 run - Section 4.4.6

use "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\Data\Final_paper_data\theta_diet.dta", clear

merge 1:1 nomem_encr using "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\Data\Final_paper_data\demographics_2018.dta"

//rename variables

drop netinc
rename geslacht gender
rename leeftijd age
rename nettoink netinc
rename nettocat netcat
rename oplzon educnodiploma
rename oplmet educdiploma
rename oplcat educcat

// 5744 people matched, 4 not
drop if _merge == 2 | _merge == 1

// Create income quartiles
egen incquart = cut(netinc), group(4)
replace incquart = 4 if incquart == 3
replace incquart = 3 if incquart == 2
replace incquart = 2 if incquart == 1
replace incquart = 1 if incquart == 0
//Note that "I don't know" or "prefer not to say" are in cat 1, but we can also drop these. The more we drop the less representative sample gets.

// Create also a category for the education (low to high)

egen educlevel = cut(educcat), group (4)
replace educlevel = 4 if educlevel == 3
replace educlevel = 3 if educlevel == 2
replace educlevel = 2 if educlevel == 1
replace educlevel = 1 if educlevel == 0

//Delete variables we dont need anymore

keep nomem_encr diet theta gender age netinc netcat incquart educnodiploma educdiploma educcat educlevel

// Drop if missing values

drop if diet == "" | gender == . | age == . | incquart == . | educlevel == . | theta == .

// 36 observations deleted

//Order data
 
order nomem_encr diet theta gender age incquart educlevel netinc netcat educnodiploma educdiploma educcat 

// Label all Variables

label variable diet "Diet - Vegan or Not"

label variable theta "Personal Preference for Veg Diet"

label variable incquart "Income Quartile"

label variable educlevel "Education Level"

//Save data set
save "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\Data\Final_paper_data\theta_diet_demographics.dta", replace

*export excel "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\Data\Reasons to Eat Less Meat\theta_diet_demographics", firstrow(varlabels) replace

//Plot and save

histogram theta, normopts(lcolor(green) lwidth(thick)) name(theta)
graph export "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\Data\Reasons to Eat Less Meat\theta_dist_stata.pdf", replace

//---- Statistics

*N=5708
*veg = 69 people or 1.21%

summarize

// Merge all of this with alpha and rho for study

merge 1:1 nomem_encr using "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\Data\Final_paper_data\alpha.dta"

drop _merge

merge 1:1 nomem_encr using "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\Data\Final_paper_data\rho.dta"

keep theta alpha rho diet age gender educlevel incquart

save "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\Data\Final_paper_data\final_dataset.dta", replace

//Summary statistics to do

drop if diet == "" | alpha == . | theta == . | rho == .

// Code for pmfs alpha/rho

*See python
