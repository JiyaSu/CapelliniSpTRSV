MTXADDRESS="/home/sujiya/SpTRSV/matrix"
PROGRAMADDRESS="/home/sujiya/SpTRSV/mix"
PROGRAM="./sptrsv"
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
