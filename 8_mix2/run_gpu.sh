MTXADDRESS="/home/sujiya/SpTRSV_Gpu/matrix"
PROGRAMADDRESS="/home/sujiya/SpTRSV_Gpu/8_mix2"
PROGRAM="./main"
CSV="result.csv"
FIN="finish"

rm $CSV
cd $MTXADDRESS
a=`ls`
#echo $a

for i in $a
do
	cd $PROGRAMADDRESS
	echo $PROGRAM $MTXADDRESS/$i
	$PROGRAM $MTXADDRESS/$i
done
echo $FIN
