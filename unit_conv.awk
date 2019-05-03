BEGIN{
	lin=68.8/1000; 
	are=lin**2; 
	vol=lin**3; 
	OFS=", "
} 
/BR9/{
	$2=$2*lin; 
	$5=$5*vol; 
	$6=$6*are; 
	$7=$7*lin;
	$8=$8/lin; 
	print $0
} 
!/BR9/
