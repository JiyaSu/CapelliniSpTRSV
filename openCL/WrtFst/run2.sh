MTXADDRESS="/home/sujiya/SpTRSV/matrix"
PROGRAMADDRESS="/home/sujiya/SpTRSV/WrtFst"
PROGRAM="./sptrsv"
CSV="result.csv"
FIN="finish"

rm $CSV
cd $MTXADDRESS
a=`ls`
judge=0
#echo $a

for i in $a
do
    if [ $i = "71.mtx" -o $i = "108.mtx" ]
    then
        judge=1

#    if [ $judge -eq 1 ]
#    then
    else
	    cd $PROGRAMADDRESS
	    echo $PROGRAM $MTXADDRESS/$i
	    $PROGRAM $MTXADDRESS/$i
    fi
done
echo $FIN
