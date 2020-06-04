MTXADDRESS="/home/storage4TB/forJiya"
PROGRAMADDRESS="/home/sujiya/SpTRSV/4_cuSP-layer2"
PROGRAM="./main"
CSV="result.csv"
FIN="finish"

rm $CSV
cd $MTXADDRESS 
a=`ls`
#echo $a

for i in $a
do
	cd $MTXADDRESS/$i
	b=`ls`
	for j in $b
	do
		cd $PROGRAMADDRESS 
		echo $PROGRAM $MTXADDRESS/$i/$j
		$PROGRAM $MTXADDRESS/$i/$j
	done
done
echo $FIN
