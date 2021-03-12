MTXADDRESS="/home/sujiya/SpTRSV/matrix"
PROGRAMADDRESS="/home/sujiya/SpTRSV/SynFree_csc"
PROGRAM="./sptrsv"
CSV="result.csv"
FIN="finish"


cd $MTXADDRESS
a=`ls`
judge=0
#echo $a

for i in $a
do
    if [ $i = "109.mtx" ]
    then
        judge=1
    fi

    if [ $judge -eq 1 ]
    then
	    cd $PROGRAMADDRESS
	    echo $PROGRAM $MTXADDRESS/$i
	    $PROGRAM $MTXADDRESS/$i
    fi
done
echo $FIN
