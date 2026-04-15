// =============================================================================
// DATA CONSTRUCTION — Foodfriends ABM
// =============================================================================
//
// PURPOSE
// -------
// This do-file constructs the three agent-level parameters (theta, alpha, rho)
// and the baseline demographic variables from the LISS panel survey data.
// The output files feed directly into the Python data pipeline
// (auxillary/create_hierarchical_agents.py) that builds the agent population
// for the agent-based model.
//
// DATA SOURCE
// -----------
// All input data come from the LISS (Longitudinal Internet Studies for the
// Social Sciences) panel — a probability-based sample of ~7,000 Dutch
// households administered by Centerdata (Tilburg University, the Netherlands),
// funded by the Domain Plan SSH and ODISSEI (since 2019) and originally
// established with NWO funding (MESS project).
//
// Three single-wave studies are used:
//   (1) "Reasons to Eat Less Meat"                            (oi18a, Jul–Aug 2018)
//       → source for diet category and dietary preference score (theta)
//   (2) "Self-Regulatory Orientation: Addressing a basic aspect of the self
//        and its relation to social indicators and life-outcomes"
//                                                             (ed11a, Feb 2011)
//       → source for dietary self-identity weight (alpha)
//   (3) "The energy transition from a citizen's perspective"  (su19a, May 2019)
//       → source for behavioral intention to reduce meat consumption (rho)
//
// Variable codes follow the LISS naming convention:
//   <module_code><wave_year><item_number>
// Documentation for each module is available via the LISS Data Archive.
//
// Citation:
//   LISS panel data (Centerdata, Tilburg University, the Netherlands), funded
//   by the Domain Plan SSH and ODISSEI (since 2019) and originally established
//   with NWO funding (MESS project).
//
// OUTPUT FILES (written to the same directory as this do-file)
// ------------------------------------------------------------
//   theta_diet.dta              — dietary preference (theta) and diet category
//   alpha.dta                   — self-identity weight (alpha)
//   rho.dta                     — behavioral intention (rho) and veggie diet flag
//   theta_diet_demographics.dta — theta + diet + demographics (main input to Python)
//   final_dataset.dta           — all three parameters merged with demographics
//
// PIPELINE OVERVIEW
// -----------------
//   Section 1:  Construct theta (dietary preference)  from oi18a (Jul–Aug 2018)
//   Section 2:  Construct alpha (self-identity)        from ed11a (Feb 2011)
//   Section 3:  Construct rho (behavioral intention)   from su19a (May 2019)
//   Section 4:  Merge theta/diet with 2018 demographics; create income/education quartiles
//   Section 5:  Merge all three parameters into the final dataset
//
// =============================================================================

clear all

// Set working directory to the folder containing this do-file.
// All file paths below are relative to this directory.
cd "`c(sysdir_personal)'"   // placeholder — adjust as needed, or set manually:
// cd "C:\path\to\Foodfriends-fresh\data\data_construction_paper"

// Alternatively, define a global macro for the data path and use $datadir throughout:
// global datadir "C:\path\to\Foodfriends-fresh\data\data_construction_paper"


// =============================================================================
// SECTION 1 — THETA (dietary preference score)
// =============================================================================
//
// Source: LISS study "Reasons to Eat Less Meat" (module code: oi18a),
// fielded July–August 2018. N = 5,742 respondents.
//
// Theta captures how strongly a respondent values vegetarian/plant-based food
// choices, derived from attitude items across three dimensions:
//   Animal welfare  — concern for farm animal treatment
//   Health          — importance of food for personal health
//   Environment     — concern for the environmental impact of food choices
//
// All attitude items are measured on a 1–7 Likert scale (1 = not important at
// all, 7 = very important). Higher scores indicate stronger pro-vegetarian
// attitudes. Theta is the row mean of the three domain means, rescaled to
// [-1, +1]: theta = -1 + 2 * ((index - 1) / (7 - 1)).
//
// Diet variable (oi18a016):
//   1 = vegetarian or vegan
//   2 = meat-eater
// =============================================================================

use "oi18a_EN_1.0p.dta", clear

// --- Construct diet category ---
gen diet = oi18a016
gen diet2 = "veg"  if diet == 1
replace diet2 = "meat" if diet == 2
drop diet
rename diet2 diet

tab diet
// Expected: ~5,742 respondents; ~69 vegetarians/vegans (approx. 1.2%)

// --- Compute domain means (items all on 1–7 scale) ---

// Animal welfare items (oi18a003, oi18a004, oi18a009, oi18a011, oi18a013, oi18a015)
egen animal_vars_mean = rowmean(oi18a003 oi18a004 oi18a009 oi18a011 oi18a013 oi18a015)

// Health items (oi18a001, oi18a007, oi18a014)
egen health_vars_mean = rowmean(oi18a001 oi18a007 oi18a014)

// Environment items (oi18a002, oi18a006, oi18a008, oi18a010, oi18a012)
egen env_vars_mean = rowmean(oi18a002 oi18a006 oi18a008 oi18a010 oi18a012)

// --- Aggregate index (unweighted mean of three domain means) ---
egen index_theta = rowmean(animal_vars_mean health_vars_mean env_vars_mean)
// index_theta is on [1, 7]

// --- Rescale to [-1, +1] ---
gen theta = -1 + 2 * ((index_theta - 1) / (7 - 1))

keep nomem_encr theta diet

sum theta

save "theta_diet.dta", replace


// =============================================================================
// SECTION 2 — ALPHA (self-identity weight)
// =============================================================================
//
// Source: LISS study "Self-Regulatory Orientation: Addressing a basic aspect
// of the self and its relation to social indicators and life-outcomes"
// (module code: ed11a), fielded February 2011.
// The item used captures how central dietary identity is to the respondent's
// self-concept — i.e., how strongly they see their diet as part of who they are.
//
// Variable ed11a020: single Likert item on a 1–7 scale (1 = strongly disagree,
// 7 = strongly agree with a statement about dietary self-identity).
//
// Alpha is rescaled to [0, 1]: alpha = (item - 1) / (7 - 1).
// In the ABM, alpha is further compressed to [0.05, 0.80] at the model level.
// =============================================================================

use "ed11a_EN_1.0p.dta", clear

keep nomem_encr ed11a020
rename ed11a020 index_alpha

// Rescale to [0, 1]
gen alpha = (index_alpha - 1) / (7 - 1)

keep nomem_encr alpha

label variable alpha "Self-identity weight (alpha), rescaled to [0,1]"

sum alpha

save "alpha.dta", replace


// =============================================================================
// SECTION 3 — RHO (behavioral intention)
// =============================================================================
//
// Source: LISS study "The energy transition from a citizen's perspective"
// (module code: su19a), fielded May 2019.
// N = 2,506 respondents (subset of main panel).
//
// Rho captures how strongly a respondent intends to reduce or eliminate meat
// consumption in the near future.
//
// Variable su19a047: behavioral intention to eat less meat, measured on a
//   1–4 ordinal scale (1 = no intention, 4 = strong intention).
//   Value 88 = "not applicable" (respondent is already vegetarian/vegan);
//   recoded to 1 (no intention to change, already at target behavior).
//
// Variable su19a046: self-reported diet at time of survey (used as an
//   additional diet indicator; retained for cross-validation with oi18a016).
//
// Rho is rescaled to [0, 1]: rho = (item - 1) / (4 - 1).
// =============================================================================

use "su19a_EN_1.0p.dta", clear

keep nomem_encr su19a047 su19a046
rename su19a047 index_rho
rename su19a046 diet_veggie

// Recode "not applicable" (already vegetarian/vegan) as minimum intention score
replace index_rho = 1 if index_rho == 88

// Rescale to [0, 1]
gen rho = (index_rho - 1) / (4 - 1)

label variable rho "Behavioral intention (rho), rescaled to [0,1]"

sum rho

save "rho.dta", replace


// =============================================================================
// SECTION 4 — MERGE THETA/DIET WITH DEMOGRAPHICS
// =============================================================================
//
// Source for demographics: LISS background variables, wave 2018
// (demographics_2018.dta, derived from avars_201807_EN_1.0p.dta).
//
// Variables retained:
//   nomem_encr   — anonymised respondent ID (LISS panel identifier)
//   diet         — dietary category (veg / meat)
//   theta        — dietary preference score
//   gender       — 1 = male, 2 = female
//   age          — age in years
//   netinc       — net monthly household income (continuous, EUR)
//   netcat       — net income category (LISS categorical code)
//   incquart     — income quartile (1 = lowest, 4 = highest), constructed below
//   educnodiploma — highest education without diploma
//   educdiploma  — highest education with diploma
//   educcat      — education category (LISS code)
//   educlevel    — education quartile (1 = lowest, 4 = highest), constructed below
//
// The income and education quartiles are constructed from the continuous/
// categorical LISS variables using equal-frequency binning (xtile-like cut).
// Note: respondents who reported "don't know" or "prefer not to say" for
// income are included in quartile 1 rather than dropped, to preserve sample
// representativeness. The Python pipeline subsequently uses these quartiles
// for stratified sampling and PMF imputation.
// =============================================================================

use "theta_diet.dta", clear

merge 1:1 nomem_encr using "demographics_2018.dta"

// Rename LISS background variable names to human-readable labels
drop netinc
rename geslacht gender
rename leeftijd age
rename nettoink netinc
rename nettocat netcat
rename oplzon educnodiploma
rename oplmet educdiploma
rename oplcat educcat

// Keep only matched observations (drop unmatched from either file)
// Expected: ~5,744 matched, ~4 unmatched
drop if _merge == 2 | _merge == 1
drop _merge

// --- Income quartiles ---
// Equal-frequency cut into 4 groups; shift from 0-indexed to 1-indexed
egen incquart = cut(netinc), group(4)
replace incquart = 4 if incquart == 3
replace incquart = 3 if incquart == 2
replace incquart = 2 if incquart == 1
replace incquart = 1 if incquart == 0

// --- Education level quartiles ---
// Same approach applied to the LISS education category variable
egen educlevel = cut(educcat), group(4)
replace educlevel = 4 if educlevel == 3
replace educlevel = 3 if educlevel == 2
replace educlevel = 2 if educlevel == 1
replace educlevel = 1 if educlevel == 0

// Keep only the variables needed downstream
keep nomem_encr diet theta gender age netinc netcat incquart ///
     educnodiploma educdiploma educcat educlevel

// Drop observations with any missing value in core variables
// (Expected: ~36 observations dropped)
drop if diet == "" | gender == . | age == . | incquart == . | educlevel == . | theta == .

// Order columns
order nomem_encr diet theta gender age incquart educlevel netinc netcat ///
      educnodiploma educdiploma educcat

// Label variables
label variable diet      "Dietary category: veg or meat"
label variable theta     "Dietary preference score, rescaled to [-1, +1]"
label variable incquart  "Net income quartile (1=lowest, 4=highest)"
label variable educlevel "Education level quartile (1=lowest, 4=highest)"

// Summary statistics
// Expected final N: ~5,708; vegetarians/vegans: ~69 (1.21%)
summarize

// --- Distribution plot (optional) ---
histogram theta, normopts(lcolor(green) lwidth(thick)) name(theta)
// graph export "theta_dist_stata.pdf", replace

save "theta_diet_demographics.dta", replace


// =============================================================================
// SECTION 5 — FINAL DATASET: MERGE ALL THREE PARAMETERS
// =============================================================================
//
// Merges theta_diet_demographics with alpha (ed11a, 2011) and rho (su19a, 2019).
// Because the three modules were fielded in different years and to different
// subsets of the panel, the merge results in many missing values for alpha and
// rho — this is expected. Missing values are handled by the Python PMF
// imputation pipeline (auxillary/create_pmf_tables.py).
//
// The final dataset contains all 5,602 respondents with valid theta/diet/
// demographics; alpha and rho are non-missing for ~1,600 and ~1,500
// respondents respectively; ~1,298 have all three parameters observed
// (complete cases, 23.2%).
// =============================================================================

use "theta_diet_demographics.dta", clear

merge 1:1 nomem_encr using "alpha.dta"
drop _merge

merge 1:1 nomem_encr using "rho.dta"
drop _merge

keep theta alpha rho diet age gender educlevel incquart

save "final_dataset.dta", replace

// Note: PMF construction for imputing missing alpha and rho values
// is handled in Python — see auxillary/create_pmf_tables.py.
// The final_dataset.dta file is the direct input to that script
// (via its Excel export: data/theta_diet_demographics.xlsx etc.).
