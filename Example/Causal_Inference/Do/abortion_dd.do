* DD estimate of 15-19 year olds in repeal states vs Roe states	
use https://github.com/scunning1975/mixtape/raw/master/abortion.dta, clear
xi: reg lnr i.repeal*i.year i.fip acc ir pi alcohol crack poverty income ur if bf15==1 [aweight=totpop], cluster(fip)

* ssc install parmest, replace
	
parmest, label for(estimate min95 max95 %8.2f) li(parm label estimate min95 max95) saving(bf15_DD.dta, replace)

use ./bf15_DD.dta, replace

keep in 17/31

gen		year=1986 in 1
replace year=1987 in 2
replace year=1988 in 3
replace year=1989 in 4
replace year=1990 in 5
replace year=1991 in 6
replace year=1992 in 7
replace year=1993 in 8
replace year=1994 in 9
replace year=1995 in 10
replace year=1996 in 11
replace year=1997 in 12
replace year=1998 in 13
replace year=1999 in 14
replace year=2000 in 15

sort year

twoway (scatter estimate year, mlabel(year) mlabsize(vsmall) msize(tiny)) (rcap min95 max95 year, msize(vsmall)), ytitle(Repeal x year estimated coefficient) yscale(titlegap(2)) yline(0, lwidth(vvvthin) lcolor(black)) xtitle(Year) xline(1986 1987 1988 1989 1990 1991 1992, lwidth(vvvthick) lpattern(solid) lcolor(ltblue)) xscale(titlegap(2)) title(Estimated effect of abortion legalization on gonorrhea) subtitle(Black females 15-19 year-olds) note(Whisker plots are estimated coefficients of DD estimator from Column b of Table 2.) legend(off)
