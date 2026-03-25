blabla //line to cut the script

use "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\foodfriends-main\demographics_2008.dta"

keep nomem_encr nohouse_encr wave geslacht gebjaar leeftijd brutoink nettoink brutocat nettocat oplzon oplmet oplcat

//rename variables

rename geslacht gender
rename gebjaar yobirth
rename leeftijd age
rename brutoink brutinc
rename nettoink netinc
rename brutocat brutcat
rename nettocat netcat
rename oplzon educnodiploma
rename oplmet educdiploma
rename oplcat educcat

save "C:\Users\emma.thill\Dropbox\Projects\Foodfriends\foodfriends-main\demographics_2008.dta", replace